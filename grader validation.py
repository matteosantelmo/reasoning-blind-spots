import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from inspect_ai import eval as inspect_eval
from omegaconf import DictConfig, OmegaConf
import json
import os
import asyncio
import re

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Scorer, accuracy, scorer, stderr

GRADER_PROMPT_TEMPLATE = """
You are an expert grader. Your task is to evaluate the correctness of a submitted answer based on the provided question and ground truth answer defining grading criteria.

---

# [Question]
{question}

---

# [Submission]
{answer}

---

# [Ground Truth]
{criterion}

---

After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes) where LETTER is either of C or I. Please choose ONE option for the grade: either "C" for correct answers, or "I" for incorrect answers. No intermediate grades are allowed. If the grading criterion is met only partially, use your best judgement to assign the most appropriate grade.

Start by carefully analyzing the submission and compare it against the ground truth. The ground truth will provide you with the necessary information to determine if the submission is correct or incorrect.
First, write a step by step reasoning about the grading criterion to make sure your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, once you have reached a final judgment, end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is either of C or I.
"""

DEFAULT_GRADE_PATTERN = r"(?i)GRADE\s*:\s*([CI])(.*)$"


"""
Evaluate a set of graders for solver given the human grading of the solver
Returns the binary metric for the accuracy of the evaluation
"""
def compute_metric(grader_scores,human_scores):
    l = len(grader_scores)
    TP=FP=TN=FN=0
    for i in range(l):
        if grader_scores[i]=="C" and human_scores[i]=="C":
            TP+=1
        elif grader_scores[i]=="C" and human_scores[i]=="I":
            FP+=1
        elif grader_scores[i]=="I" and human_scores[i]=="C":
            FN+=1
        elif grader_scores[i]=="I" and human_scores[i]=="I":
            TN +=1

    accuracy = (TP+TN)/l
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    FPR = FP/(FP+TN) if (FP+TN)>0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "FPR": FPR
    }
    
"""
The evaluation.jsonl file contains the prompt, groundtruth answer, solver answer, solver model and human grading. 
This is passed with the Config for the grader. It returns the evaluation of the grader.
"""
async def evaluation(cfg: DictConfig, val_path):
    grader_scores = []
    human_scores = []
    grader = grader_score(cfg.grader)
    with open(val_path,"r") as f:
        for line in f:
            sample = json.loads(line)
            prompt = sample["prompt"]
            solver_answer = sample["solver_answer"]
            solution = sample["solution"]
            human_score = sample["human_score"]
            human_scores.append(human_score)
            score = await grader(prompt,solver_answer,solution)
            grader_scores.append(score.value)

    return compute_metric(grader_scores,human_scores)

"Need to write this grader_score such that it takes the prompt,solver and solution and return C or I"

@scorer(metrics=[accuracy(), stderr()])
def model_grading(
    grader_model: Model,
    template: str = GRADER_PROMPT_TEMPLATE,
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
) -> Scorer:

    async def score(prompt: str, ground_truth: str, solver_answer: str) -> Score:
        clean_answer = re.sub(r"</?think>", "", solver_answer).strip()
        if len(clean_answer) == 0:
            raise ValueError("The cleaned answer is empty. Raw answer:\n" + solver_answer)

        # Format the grading prompt with the cleaned answer
        score_prompt = template.format(
            question= prompt,
            answer=solver_answer,
            criterion=ground_truth,
        )
        metadata = {
            "raw_answer": solver_answer,
            "thinking_stripped": solver_answer != clean_answer,
        }

        # Query the grader model
        result = await grader_model.generate(score_prompt)

        # Extract the grade from the response
        match = re.search(grade_pattern, result.completion, re.MULTILINE | re.DOTALL)
        if match:
            grade = match.group(1).upper()
            return Score(
                value=grade,
                answer=clean_answer,
                explanation=result.completion,
                metadata=metadata,
            )
        else:
            return Score(
                value="I",
                answer=clean_answer,
                explanation=f"Grade not found in model output: {result.completion}",
                metadata=metadata,
            )

    return score


def grader_score(
    grader_config: dict = None,
) -> Scorer:
    """
    Returns a model-graded scorer (grader) using the specified model.

    This grader automatically strips <think>...</think> reasoning traces from
    model outputs before evaluation, ensuring only the final answer is graded.
    """

    if "backend" not in grader_config or "model_name" not in grader_config:
        raise ValueError("grader_config must contain 'backend' and 'model_name' keys.")

    model_str = grader_config.backend + "/" + grader_config.model_name
    gen_config = grader_config.get("generate_config", {})

    # Any other argument in the config is passed to the model
    model_args = {
        k: v
        for k, v in grader_config.items()
        if k not in ["backend", "model_name", "generate_config"]
    }

    grader = get_model(
        model=model_str,
        role="grader",
        config=GenerateConfig(**gen_config),
        **model_args,
    )

    return model_grading(
        grader_model=grader,
        template=GRADER_PROMPT_TEMPLATE,
        grade_pattern=DEFAULT_GRADE_PATTERN,
    )
