import base64
import io
import json
import os

import datasets
from hydra.utils import get_original_cwd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from omegaconf import DictConfig
from PIL import Image as PILImage


def pil_image_to_base64_url(image: PILImage.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_dataset(cfg: DictConfig) -> MemoryDataset:
    """
    Loads the dataset from a JSONL file or Hugging Face Hub and converts it to Inspect Samples.
    Handles multimodal inputs by checking the 'modality' field.
    NOTE: image-gen is not supported yet.

    Args:
        cfg: Configuration dictionary containing dataset parameters.

    Returns:
        MemoryDataset: The loaded dataset as a MemoryDataset object.
    """
    if "question_type" in cfg:
        allowed_types = set(cfg.question_type)
    else:
        allowed_types = {"text-only"}
        print("No question_type specified in config; defaulting to 'text-only'.")

    local_path = os.path.join(get_original_cwd(), cfg.path)

    samples = []
    if os.path.exists(local_path):
        print(f"Loading local dataset from: {local_path}")
        with open(local_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                question_type = record.get("question_type")
                if not question_type or question_type not in allowed_types:
                    continue

                # Resolve image paths if present
                if "image" in record and record["image"] is not None:
                    record["image_source"] = os.path.join(
                        get_original_cwd(), record["image"]
                    )

                samples.append(create_sample(record, question_type))

    else:
        print(f"Loading dataset from Hugging Face Hub: {cfg.path}")
        try:
            split = cfg.get("split", "test")
            ds = datasets.load_dataset(cfg.path, split=split)

            for record in ds:
                question_type = record.get("question_type")
                if not question_type or question_type not in allowed_types:
                    continue

                # Normalize record for HF dataset
                if "QID" in record:
                    record["index"] = record["QID"]

                samples.append(create_sample(record, question_type))

        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from '{cfg.path}' (checked local path '{local_path}'): {e}"
            )

    if not samples:
        raise ValueError(
            "No samples loaded from the dataset. Please check the dataset file and configuration."
        )

    if cfg.get("limit") is not None:
        samples = samples[: cfg.limit]

    return MemoryDataset(samples)


def create_sample(record: dict, question_type: str) -> Sample:
    """Helper to create a Sample from a record dictionary."""
    prompt = record["prompt"]
    input_content = None

    if question_type == "text-only":
        input_content = prompt
    elif question_type in ["multi-to-text", "image-to-text"]:
        if "image" not in record or record["image"] is None:
            raise ValueError(
                f"Record {record.get('index')} missing 'image' field for multimodal question."
            )

        image_content = None

        # Check if it is a local path (string)
        if "image_source" in record:
            image_content = record["image_source"]
        # Check if it is a PIL Image
        elif isinstance(record["image"], PILImage.Image):
            image_content = pil_image_to_base64_url(record["image"])
        # Check if it is a string (e.g. path or base64)
        elif isinstance(record["image"], str):
            image_content = record["image"]
        else:
            raise ValueError(f"Unknown image format in record {record.get('index')}")

        input_content = [
            ChatMessageUser(
                content=[
                    ContentText(text=prompt),
                    ContentImage(image=image_content),
                ]
            )
        ]
    else:
        raise ValueError(f"Unsupported question_type: {question_type}")

    return Sample(
        input=input_content,
        target=record["solution"],
        id=record["index"],
        metadata={
            "question_type": question_type,
            "prompt": prompt,
            "categories": record.get("categories", ""),
            "failure_modes": record.get("failure_modes", ""),
        },
    )
