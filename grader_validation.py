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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("grader_validation")


def compute_metrics(
    grader_scores: List[str], human_scores: List[str]
) -> Dict[str, float]:
    """
    Compare grader scores with human provided ground truth scores,
    and compute binary classification metrics.

    Args:
        grader_scores (List[str]): List of scores from the grader ("C" or "I").
        human_scores (List[str]): List of ground truth scores from humans ("C" or "I").
    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, and FPR.
    """
    l = len(grader_scores)
    tp = fp = tn = fn = 0
    for i in range(l):
        if grader_scores[i] == "C" and human_scores[i] == "C":
            tp += 1
        elif grader_scores[i] == "C" and human_scores[i] == "I":
            fp += 1
        elif grader_scores[i] == "I" and human_scores[i] == "C":
            fn += 1
        elif grader_scores[i] == "I" and human_scores[i] == "I":
            tn += 1

    accuracy = (tp + tn) / l
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "FPR": fpr}


async def grader_validation(cfg: DictConfig) -> List[Dict]:
    """
    Reads the validation dataset and evaluates the defined grader on it.
    The configuration of the grader is passed with the cfg argument.

    Args:
        cfg (DictConfig): Configuration containing grader settings and dataset path.

    Returns:
        List[Dict]: List of dictionaries containing the evaluation results for each sample.
    """
    results = []
    grader = get_grader(cfg.grader, str_input=True)
    val_path = cfg.dataset.path
    limit = cfg.dataset.get("limit", None)

    with open(val_path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            sample = json.loads(line)
            prompt = sample["prompt"]
            solver_answer = sample["solver_answer"]
            solution = sample["solution"]
            human_score = sample["human_grade"]
            image = sample.get("image")

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
                "solution": solution,
                "human_score": human_score,
                "grader_score": score.value,
                "grader_explanation": score.explanation,
                "usage": score.metadata.get("usage", {}),
            }
            results.append(sample_result)

            if sample_result["usage"]:
                logger.info(
                    f"Sample {i}: Grader Score={sample_result['grader_score']},\tHuman Score={sample_result['human_score']}\t- "
                    f"Input Tokens={sample_result['usage'].get('input_tokens', 0)},\t"
                    f"Reasoning Tokens={sample_result['usage'].get('reasoning_tokens', 0)},\t"
                    f"Output Tokens={sample_result['usage'].get('output_tokens', 0)}\t"
                )

    return results


@hydra.main(config_path="conf", config_name="config_val", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for the grader validation script.
    Runs validation, saves results to a JSONL file, and prints metrics.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    results = asyncio.run(grader_validation(cfg))

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
    output_file_path = os.path.join(output_dir, "validation_results.jsonl")

    with open(output_file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Extract scores for metric computation
    g_scores = [r["grader_score"] for r in results]
    h_scores = [r["human_score"] for r in results]

    # Compute and print metrics
    metrics = compute_metrics(g_scores, h_scores)
    logger.info(
        "Grader Validation Metrics: {}".format(
            ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
    )


if __name__ == "__main__":
    main()
