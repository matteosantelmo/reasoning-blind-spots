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
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import code_execution
from inspect_ai.util import message_limit

# Prompt template for grading image generation tasks
IMAGE_GRADER_PROMPT_TEMPLATE = """
You are an expert grader evaluating whether a generated image correctly satisfies the given task.

---

# [Task/Prompt]
{question}

{input_image_section}

---

# [Generated Image (Candidate Answer to Grade)]
The image below is the generated output that needs to be evaluated.
(The generated image is attached after this text)

---

# [Grading Criteria]
{criterion}

---

# [Instructions]
Evaluate whether the generated image correctly satisfies the task requirements based on the grading criteria above.

Consider the following aspects when grading:
1. **Content Accuracy**: Does the image contain the required elements/objects/scenes?
2. **Instruction Following**: Does the image follow the specific instructions in the prompt?
3. **Quality**: Is the generated image of reasonable quality (not corrupted, incomplete, or nonsensical)?
4. **Relevance**: Is the generated image relevant to the task?

If needed, you can execute Python code snippets to help verify aspects of the generated image (e.g., counting objects, checking colors, analyzing dimensions, etc.).

After your analysis, provide your final grade.
Reply with 'GRADE: $LETTER' (without quotes) where LETTER is either:
- "C" for CORRECT: The generated image satisfies the task requirements
- "I" for INCORRECT: The generated image does NOT satisfy the task requirements

Start with a brief analysis of the generated image compared to the requirements, then provide your final grade.
"""

# Section to add when the original question includes an input image
INPUT_IMAGE_SECTION_TEMPLATE = """
# [Input Image from Question]
The following image was provided as part of the original question (this is NOT the generated image):
The input image is attached first, before the generated image
"""

DEFAULT_GRADE_PATTERN = r"(?i)GRADE\s*:\s*([CI])(.*)$"


def get_raw_answer_for_image(state: TaskState) -> tuple[str, Optional[str]]:
    """
    Extract the model's answer and generated image path from the TaskState.

    For image generation tasks, the answer includes both a text description
    and the path to the generated image.

    Args:
        state: The current TaskState.

    Returns:
        Tuple of (text answer, generated image URL/path or None).
    """
    text_answer = ""
    image_url = None

    # Prefer data URL for proper visualization in Inspect UI
    # Fall back to file path if data URL not available
    image_url = state.store.get("generated_image_data_url")
    if not image_url:
        image_url = state.store.get("generated_image_path")

    # Get text completion
    if state.output and state.output.completion:
        text_answer = state.output.completion

    # Also check output content for image
    if state.output and state.output.choices:
        for choice in state.output.choices:
            if hasattr(choice, "message") and choice.message.content:
                for content in choice.message.content:
                    if isinstance(content, ContentImage):
                        image_url = image_url or content.image
                    elif isinstance(content, ContentText):
                        text_answer = text_answer or content.text

    return text_answer, image_url


async def score_image_generation(
    prompt: str,
    ground_truth: str,
    generated_image_path: str,
    grader_model: Model,
    input_image: Optional[str] = None,
    template: str = IMAGE_GRADER_PROMPT_TEMPLATE,
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
) -> Score:
    """
    Score an image generation task.

    This function prepares the grading prompt with proper distinction between
    input images (from question) and generated images (solver output), then
    queries the grader model (a VLM) to evaluate the result.

    Args:
        prompt: The original question/prompt for image generation.
        ground_truth: The grading criteria (what makes the image correct).
        generated_image_path: Path or URL to the generated image.
        grader_model: The VLM model to use for grading.
        input_image: Optional path/URL to input image from the original question.
        template: The grading prompt template.
        grade_pattern: Regex pattern to extract the grade.

    Returns:
        Score object with grade (C/I), explanation, and metadata.
    """
    metadata = {
        "generated_image_path": generated_image_path,
        "has_input_image": input_image is not None,
    }

    # Build the grading prompt
    input_image_section = ""
    if input_image:
        input_image_section = INPUT_IMAGE_SECTION_TEMPLATE

    score_prompt = template.format(
        question=prompt,
        input_image_section=input_image_section,
        criterion=ground_truth,
    )

    # Build content list with images in correct order
    content_list = [ContentText(text=score_prompt)]
    if input_image:
        content_list.append(ContentImage(image=input_image))
    content_list.append(ContentImage(image=generated_image_path))

    # Query the grader in a generation loop to allow tool use (code execution)
    grader_input = [ChatMessageUser(content=content_list)]
    max_grader_messages = 5
    try:
        with message_limit(max_grader_messages):
            _, result = await grader_model.generate_loop(
                grader_input, tools=[code_execution()]
            )
    except Exception as e:
        return Score(
            value="I",
            answer=f"[Generated Image: {generated_image_path}]",
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
            answer=f"[Generated Image: {generated_image_path}]",
            explanation=result.completion,
            metadata=metadata,
        )
    else:
        return Score(
            value="I",
            answer=f"[Generated Image: {generated_image_path}]",
            explanation=f"Grade not found in model output: {result.completion}",
            metadata=metadata,
        )


@scorer(metrics=[accuracy()])
def image_generation_grader(
    grader_model: Model,
    template: str = IMAGE_GRADER_PROMPT_TEMPLATE,
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
) -> Scorer:
    """
    Custom scorer for image generation tasks.

    This scorer evaluates generated images against the task requirements
    using a VLM as the grader.

    Args:
        grader_model: The VLM model to use for grading.
        template: The grading prompt template.
        grade_pattern: Regex pattern to extract the grade.

    Returns:
        Scorer function compatible with Inspect AI.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the generated image path
        text_answer, generated_image_path = get_raw_answer_for_image(state)

        # Check if generation failed
        generation_error = state.store.get("generation_error")
        if generation_error:
            return Score(
                value="I",
                answer=text_answer,
                explanation=f"Image generation failed: {generation_error}",
                metadata={"generation_error": generation_error},
            )

        if not generated_image_path:
            return Score(
                value="I",
                answer=text_answer,
                explanation="No generated image found in task state.",
                metadata={},
            )

        # Extract input image from the original question (if any)
        input_image = None
        for message in state.messages:
            if message.role == "user" and isinstance(message.content, list):
                for content in message.content:
                    if isinstance(content, ContentImage):
                        input_image = content.image
                        break
            if input_image:
                break

        # Score the generated image
        return await score_image_generation(
            prompt=state.input_text,
            ground_truth=target.text,
            generated_image_path=generated_image_path,
            grader_model=grader_model,
            input_image=input_image,
            template=template,
            grade_pattern=grade_pattern,
        )

    return score


def get_image_grader(grader_config: dict, use_pregenerated_input: bool = False):
    """
    Factory function to create an image generation grader from configuration.

    Args:
        grader_config: Configuration dictionary containing:
            - backend: Model backend (e.g., "google", "openai")
            - model_name: Model name (must be a VLM)
            - generate_config: Optional generation configuration
        use_pregenerated_input: If True, returns a callable that takes inputs directly

    Returns:
        Configured image generation grader (Scorer) or callable if use_pregenerated_input=True.
    """
    if "backend" not in grader_config or "model_name" not in grader_config:
        raise ValueError("grader_config must contain 'backend' and 'model_name' keys.")

    model_str = grader_config["backend"] + "/" + grader_config["model_name"]
    gen_config = grader_config.get("generate_config", {})

    # Any other argument in the config is passed to the model
    model_args = {
        k: v
        for k, v in grader_config.items()
        if k not in ["backend", "model_name", "generate_config", "enabled"]
    }

    grader_model = get_model(
        model=model_str,
        role="grader",
        config=GenerateConfig(**gen_config),
        **model_args,
    )

    if not use_pregenerated_input:
        # Return the Scorer function to be used with a solver model
        return image_generation_grader(
            grader_model=grader_model,
            template=IMAGE_GRADER_PROMPT_TEMPLATE,
            grade_pattern=DEFAULT_GRADE_PATTERN,
        )
    else:
        # Return a callable that takes pregenerated inputs
        async def grader_str(
            prompt: str,
            ground_truth: str,
            generated_image_path: str,
            input_image: Optional[str] = None,
        ):
            return await score_image_generation(
                prompt=prompt,
                ground_truth=ground_truth,
                generated_image_path=generated_image_path,
                grader_model=grader_model,
                input_image=input_image,
                template=IMAGE_GRADER_PROMPT_TEMPLATE,
                grade_pattern=DEFAULT_GRADE_PATTERN,
            )

        return grader_str
