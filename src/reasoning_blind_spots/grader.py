import re
from typing import Optional

from inspect_ai.model import (
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import code_execution
from inspect_ai.util import message_limit

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

After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes) where LETTER is either of C or I. Please choose ONE option for the grade: either "C" for correct answers, or "I" for incorrect answers. No intermediate grades are allowed. If the grading criterion is met only partially or the ground truth is ambiguous, use your best judgement to assign the most appropriate grade.

Start by briefly analyzing the submission and compare it against the ground truth. The ground truth will provide you with the necessary information to determine if the submission is correct or incorrect. If needed, execute code snippets to verify the correctness of the submission.
Write a minimal explanation of your reasoning. Then, once you have reached a final judgment, end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is either of C or I.
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

    Attempts to get the completion string directly. If empty, falls back
    to joining text content from choices.

    Args:
        state: The current TaskState.

    Returns:
        str: The extracted answer string.
    """
    # First try to get the answer from the 'completion' directly
    answer = state.output.completion
    if len(answer) > 0:
        return answer

    try:
        # Fallback: try to reconstruct the answer from the content list
        answer_parts = []
        for content in state.output.choices[0].message.content:
            if content.text and len(content.text) > 0:
                answer_parts.append(content.text)
            else:
                # Unknown or empty content
                continue

        return "\n".join(answer_parts).strip()
    except Exception as e:
        return ""


async def score_str(
    prompt: str,
    ground_truth: str,
    solver_answer: str,
    grader_model: Model,
    image: Optional[str] = None,
    template: str = GRADER_PROMPT_TEMPLATE,
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
) -> Score:
    """
    Grading logic that takes raw strings as input and returns a Score object.

    This function handles stripping thinking tags, formatting the prompt,
    querying the grader model, and parsing the result.

    Args:
        prompt: The input question/prompt.
        ground_truth: The ground truth answer/criterion.
        solver_answer: The answer provided by the solver model.
        grader_model: The model to use for grading.
        image: Optional path or url to an image to be included in the prompt.
        template: The grading prompt template.
        grade_pattern: Regex pattern to extract the grade.

    Returns:
        Score: A Score object containing the grade (C/I), explanation, and metadata.

    Raises:
        ValueError: If the cleaned answer is empty.
    """

    clean_answer = strip_thinking_tags(solver_answer)
    if len(clean_answer) == 0:
        raise ValueError("The cleaned answer is empty. Raw answer:\n" + solver_answer)

    # Format the grading prompt with the cleaned answer
    score_prompt = template.format(
        question=prompt,
        answer=clean_answer,
        criterion=ground_truth,
    )
    metadata = {
        "raw_answer": solver_answer,
        "thinking_stripped": solver_answer != clean_answer,
    }

    # Prepare the input message(s) for the grader model
    if image:
        loop_input = [
            ChatMessageUser(
                content=[ContentText(text=score_prompt), ContentImage(image=image)]
            )
        ]
    else:
        loop_input = score_prompt

    # Query the grader in a generation loop to allow tool use (code execution)
    max_grader_messages = 5
    try:
        with message_limit(max_grader_messages):
            _, result = await grader_model.generate_loop(
                loop_input, tools=[code_execution()]
            )
    except Exception as e:
        return Score(
            value="I",
            answer=clean_answer,
            explanation=f"Grader error (possibly exceeded message limit): {str(e)}",
            metadata=metadata,
        )

    if result.usage:
        metadata["usage"] = {
            "input_tokens": result.usage.input_tokens,
            "reasoning_tokens": (
                result.usage.reasoning_tokens
                if result.usage.reasoning_tokens is not None
                else 0
            ),
            "output_tokens": result.usage.output_tokens,
        }
        metadata["total_tokens"] = sum(metadata["usage"].values())

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


@scorer(metrics=[accuracy()])
def model_graded_qa_with_reasoning_stripped(
    grader_model: Model,
    template: str = GRADER_PROMPT_TEMPLATE,
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
) -> Scorer:
    """
    Custom scorer that enforces thinking/reasoning stripping before grading.

    Args:
        grader_model: The Model instance to use for grading.
        template: The grading prompt template.
        grade_pattern: Regex pattern to extract the grade from grader output.

    Returns:
        Scorer: A Scorer function compatible with inspect_ai.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's completion and strip any thinking traces
        raw_answer = get_raw_answer(state)

        # Extract image from state.messages
        image = None
        for message in state.messages:
            if message.role == "user" and isinstance(message.content, list):
                for content in message.content:
                    if isinstance(content, ContentImage):
                        image = content.image
                        break
            if image:
                break

        return await score_str(
            state.input_text,
            target.text,
            raw_answer,
            grader_model,
            image=image,
            template=template,
            grade_pattern=grade_pattern,
        )

    return score


def get_grader(grader_config: dict = None, str_input=False):
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
        if k not in ["backend", "model_name", "generate_config", "enabled"]
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

        async def grader_str(
            prompt: str,
            ground_truth: str,
            solver_answer: str,
            image: Optional[str] = None,
        ):
            return await score_str(
                prompt,
                ground_truth,
                solver_answer,
                grader_model=grader,
                image=image,
                template=GRADER_PROMPT_TEMPLATE,
                grade_pattern=DEFAULT_GRADE_PATTERN,
            )

        return grader_str
