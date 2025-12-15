from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Scorer, model_graded_qa

GRADER_PROMPT_TEMPLATE = """
You are an expert grader. Your task is to evaluate the correctness of a submitted answer based on the provided question and ground truth answer (criterion).

---

## Question
{question}

## Submission
{answer}

## Ground Truth
{criterion}

---

After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes) where LETTER is either of C or I. Please choose ONE option for the grade: either "C" for correct answers, or "I" for incorrect answers. No intermediate grades are allowed.

Start by carefully analyzing the submission and compare it against the ground truth. The ground truth will provide you with the necessary information to determine if the submission is correct or incorrect.
First, write a step by step reasoning about the grading criterion to make sure your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, once you have reached a final judgment, end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is either of C or I.
"""

DEFAULT_GRADE_PATTERN = r"(?i)GRADE\s*:\s*([CI])(.*)$"


def get_grader(
    grader_config: dict = None,
) -> Scorer:
    """
    Returns a model-graded scorer (grader) using the specified model.
    """

    if "backend" not in grader_config or "model_name" not in grader_config:
        raise ValueError("grader_config must contain 'backend' and 'model_name' keys.")

    model_str = grader_config.backend + "/" + grader_config.model_name
    gen_config = grader_config.get("generate_config", {})

    grader = get_model(
        model=model_str, role="grader", config=GenerateConfig(**gen_config)
    )
    return model_graded_qa(
        model=grader,
        template=GRADER_PROMPT_TEMPLATE,
        grade_pattern=DEFAULT_GRADE_PATTERN,
        instructions=None,
        partial_credit=False,
    )
