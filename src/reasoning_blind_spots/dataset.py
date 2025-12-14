import json
import os
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ContentImage, ContentText, ChatMessageUser

def load_dataset(jsonl_path: str) -> MemoryDataset:
    """
    Loads the dataset from a JSONL file and converts it to Inspect Samples.
    Handles multimodal inputs by checking the 'modality' field.
    """
    samples = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Construct input based on modality
            if record.get("modality") == "multimodal" and "image" in record:
                # Ensure image path is absolute or relative to CWD
                image_path = record["image"]
                input_content = [
                    ChatMessageUser(content=[
                        ContentText(text=record["prompt"]),
                        ContentImage(image=image_path)
                    ])
                ]
            else:
                input_content = record["prompt"]
            
            samples.append(Sample(
                input=input_content,
                target=record["solution"],
                id=record["index"],
                metadata={
                    "modality": record.get("modality", "text"),
                    "prompt_text": record["prompt"] # Store original prompt text in metadata
                }
            ))
    return MemoryDataset(samples)
