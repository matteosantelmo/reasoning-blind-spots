import sys
import os
from inspect_ai import task

# Ensure src is in the path so we can import the package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from reasoning_blind_spots.task import reasoning_benchmark as rb

@task
def benchmark(dataset_path: str = "data/dummy_dataset.jsonl", verifier_model: str = "google/gemini-2.5-flash-lite"):
    """
    Wrapper task to run the reasoning benchmark from the root directory.
    """
    return rb(dataset_path=dataset_path, verifier_model=verifier_model)
