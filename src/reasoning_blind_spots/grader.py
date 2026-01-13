import re

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

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


def strip_thinking_tags(text: str) -> str:
    """
    Remove thinking/reasoning traces from model output.

    If the text contains a </think> tag, return only the content after it.
    This handles models that use <think>...</think> tags for chain-of-thought reasoning.

    Args:
        text: The model output text that may contain thinking tags.

    Returns:
        The text with thinking content removed, or the original text if no thinking tags found.
    """
    # Look for </think> and return everything after it
    think_end_match = re.search(r"</think>\s*", text, re.DOTALL)
    if think_end_match:
        completion = text[think_end_match.end() :].strip()
        if len(completion) > 0:
            return completion
        else:
            # If nothing after </think> return the original text
            return text

    return text


def get_raw_answer(state: TaskState) -> str:
    """
    Extract the model's answer from the TaskState.
    """
    # First try to get the answer from the 'completion' directly
    answer = state.output.completion
    if len(answer) > 0:
        return answer

    # Fallback: try to reconstruct the answer from the content list
    answer_parts = []
    for content in state.output.choices[0].message.content:
        if content.text and len(content.text) > 0:
            answer_parts.append(content.text)
        else:
            # Unknown or empty content
            continue

    return "\n".join(answer_parts).strip()

@scorer(metrics=[accuracy(), stderr()])
### Defining a score that takes inputs as strings
async def score_str(prompt: str, ground_truth: str, solver_answer: str, grader_model: Model, template: str = GRADER_PROMPT_TEMPLATE, grade_pattern: str = DEFAULT_GRADE_PATTERN) -> Score:
        clean_answer = strip_thinking_tags(solver_answer)
        if len(clean_answer) == 0:
            raise ValueError("The cleaned answer is empty. Raw answer:\n" + solver_answer)

        # Format the grading prompt with the cleaned answer
        score_prompt = template.format(
            question= prompt,
            answer=clean_answer,
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



def model_graded_qa_with_reasoning_stripped(
    grader_model: Model,
    template: str = GRADER_PROMPT_TEMPLATE,
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
) -> Scorer:
    """
    Custom scorer that enforces thinking/reasoning stripping before grading.

    Args:
        template: The grading prompt template.
        grade_pattern: Regex pattern to extract the grade from grader output.
        model: The model string (e.g., "openai/gpt-4") to use for grading.
        gen_config: Generation configuration for the grader model.

    Returns:
        A Scorer function.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's completion and strip any thinking traces
        raw_answer = get_raw_answer(state)

        return await score_str(state.input_text, target.text, raw_answer, grader_model, template=template, grade_pattern=grade_pattern)

    return score


def get_grader(
    grader_config: dict = None,
    str_input = False
):
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

    if str_input == False:
        return model_graded_qa_with_reasoning_stripped(
            grader_model=grader,
            template=GRADER_PROMPT_TEMPLATE,
            grade_pattern=DEFAULT_GRADE_PATTERN,
        )
    else: 
        async def grader_str(prompt: str,ground_truth: str,solver_answer: str):
            return await score_str(prompt,ground_truth,solver_answer,
                                  grader_model=grader,
                                template=GRADER_PROMPT_TEMPLATE,
                                grade_pattern=DEFAULT_GRADE_PATTERN )
        return grader_str
        