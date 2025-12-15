import json
import os

from hydra.utils import get_original_cwd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from omegaconf import DictConfig


def load_dataset(cfg: DictConfig) -> MemoryDataset:
    """
    Loads the dataset from a JSONL file and converts it to Inspect Samples.
    Handles multimodal inputs by checking the 'modality' field.
    NOTE: image-gen is not supported yet.

    Args:
        cfg: Configuration dictionary containing dataset parameters.

    Returns:
        MemoryDataset: The loaded dataset as a MemoryDataset object.
    """
    # Resolve path relative to original working directory (before Hydra changed it)
    jsonl_path = os.path.join(get_original_cwd(), cfg.path)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    if "question_type" in cfg:
        allowed_types = set(cfg.question_type)
    else:
        allowed_types = {"text-only", "multi-to-text"}
        print("No question_type specified in config; defaulting to 'text-only'.")

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            question_type = record.get("question_type")
            if not question_type or question_type not in allowed_types:
                continue

            # Prepare input based on modality
            prompt = record["prompt"]
            if question_type == "text-only":
                input_content = prompt
            elif question_type in ["multi-to-text", "image-to-text"]:
                if "image" not in record:
                    raise ValueError(
                        f"Record {record.get('index')} missing 'image' field for multimodal question."
                    )
                # Resolve image path relative to original working directory
                image_path = os.path.join(get_original_cwd(), record["image"])
                input_content = [
                    ChatMessageUser(
                        content=[
                            ContentText(text=prompt),
                            ContentImage(image=image_path),
                        ]
                    )
                ]
            else:
                raise ValueError(f"Unsupported question_type: {question_type}")

            samples.append(
                Sample(
                    input=input_content,
                    target=record["solution"],
                    id=record["index"],
                    metadata={
                        "question_type": question_type,
                        "prompt": prompt,
                    },
                )
            )
    if not samples:
        raise ValueError(
            "No samples loaded from the dataset. Please check the dataset file and configuration."
        )

    if cfg.get("limit") is not None:
        samples = samples[: cfg.limit]

    return MemoryDataset(samples)
