from inspect_ai import Task, task
from inspect_ai.solver import generate
from reasoning_blind_spots.dataset import load_dataset
from reasoning_blind_spots.scorer import get_verifier

# Default configuration values
DEFAULT_DATASET_PATH = "data/dummy_dataset.jsonl"
DEFAULT_VERIFIER_MODEL = "google/gemini-2.5-flash-lite"

@task
def reasoning_benchmark(dataset_path: str = DEFAULT_DATASET_PATH, verifier_model: str = DEFAULT_VERIFIER_MODEL):
    """
    Defines the reasoning benchmark task.
    
    Args:
        dataset_path: Path to the dataset file.
        verifier_model: Model to use for verification.
    """
    # Load the dataset
    dataset = load_dataset(dataset_path)

    # Define the verifier (scorer)
    verifier = get_verifier(model_name=verifier_model)

    return Task(
        dataset=dataset,
        plan=[
            generate()
        ],
        scorer=verifier
    )
