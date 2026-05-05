import asyncio
import json
import logging
import os
from typing import Dict, List

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from reasoning_blind_spots.grader import get_grader
from reasoning_blind_spots.image_grader import get_image_grader

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("grade_only")


async def grade_only(cfg: DictConfig) -> List[Dict]:
    """
    Reads the validation dataset and evaluates the defined grader on it.
    The configuration of the grader is passed with the cfg argument.

    Supports both text-output tasks (using the text grader) and image generation
    tasks (using the image grader). The task type is determined by the
    'question_type' field in each sample.

    Args:
        cfg (DictConfig): Configuration containing grader settings and dataset path.

    Returns:
        List[Dict]: List of dictionaries containing the evaluation results for each sample.
    """
    results = []
    val_path = cfg.dataset.path
    limit = cfg.dataset.get("limit", None)

    # Determine if this is an image generation validation based on dataset path or config
    is_image_gen = (
        "image" in val_path.lower()
        or cfg.dataset.get("task_type", "") == "image-generation"
    )

    # Initialize the appropriate grader
    if is_image_gen:
        grader = get_image_grader(cfg.grader, use_pregenerated_input=True)
    else:
        grader = get_grader(cfg.grader, use_pregenerated_input=True)

    with open(val_path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            sample = json.loads(line)
            prompt = sample["prompt"]
            # Handle both 'solver_answer' and 'solver_solution' keys (some samples use different keys)
            solver_answer = sample.get("solver_answer") or sample.get("solver_solution")
            solution = sample["solution"]
            image = sample.get("image")
            question_type = sample.get("question_type", "")

            # Use the appropriate grading method based on task type
            if question_type == "text-to-image" or is_image_gen:
                # Image generation task: solver_answer is the generated image path
                score = await grader(
                    prompt=prompt,
                    ground_truth=solution,
                    generated_image_path=solver_answer,
                    input_image=image,  # Input image from the question, if any
                )
            else:
                # Text output task
                score = await grader(
                    prompt=prompt,
                    ground_truth=solution,
                    solver_answer=solver_answer,
                    image=image,
                )

            # Store all information
            sample_result = {
                "prompt": prompt,
                "solver_answer": solver_answer,
                "index": sample.get("index", i),
                "solver_name": sample.get("solver_name", "unknown"),
                "solution": solution,
                "grader_score": score.value,
                "grader_explanation": score.explanation,
                "usage": (
                    score.metadata.get("usage", {})
                    if score.metadata is not None
                    else {}
                ),
                "question_type": question_type,
                "full_sample": sample,  # Store the full original sample for reference
            }
            results.append(sample_result)

            if sample_result["usage"]:
                logger.info(
                    f"Sample {i}: Grader Score={sample_result['grader_score']}\t- "
                    f"Input Tokens={sample_result['usage'].get('input_tokens', 0)},\t"
                    f"Reasoning Tokens={sample_result['usage'].get('reasoning_tokens', 0)},\t"
                    f"Output Tokens={sample_result['usage'].get('output_tokens', 0)}\t"
                )

    return results


@hydra.main(config_path="conf", config_name="regrade", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for the grader validation script.
    Runs validation, saves results to a JSONL file, and prints metrics.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    results = asyncio.run(grade_only(cfg))

    # Calculate total token usage
    total_input_tokens = sum(r["usage"].get("input_tokens", 0) for r in results)
    total_output_tokens = sum(
        r["usage"].get("output_tokens", 0) for r in results
    ) + sum(r["usage"].get("reasoning_tokens", 0) for r in results)

    logger.info(f"Total Input Tokens: {total_input_tokens}")
    logger.info(f"Total Output Tokens: {total_output_tokens}")
    logger.info(f"Total Token Usage: {total_input_tokens + total_output_tokens}")

    # Save results to file as JSONL
    output_dir = HydraConfig.get().runtime.output_dir
    output_file_path = os.path.join(output_dir, "regrading_outputs.jsonl")

    with open(output_file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Validation results saved to {output_file_path}")


if __name__ == "__main__":
    main()
