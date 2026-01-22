from inspect_ai import Task, task
from inspect_ai.solver import generate
from omegaconf import DictConfig, OmegaConf

from reasoning_blind_spots.dataset import load_dataset
from reasoning_blind_spots.grader import get_grader
from reasoning_blind_spots.image_grader import get_image_grader
from reasoning_blind_spots.image_solver import get_image_generation_solver
from reasoning_blind_spots.solver import get_solver


def is_image_generation_task(cfg: DictConfig) -> bool:
    if "question_type" not in cfg.dataset:
        return False

    question_types = set(cfg.dataset.question_type)
    return bool(
        question_types
        & {"text-to-image", "image-gen", "multi-to-image", "image-to-image"}
    )


@task
def reasoning_benchmark(cfg: DictConfig):
    """
    Defines the reasoning benchmark task.

    This task supports both text generation and image generation tasks:
    - For text generation tasks (text-only, multi-to-text): Uses standard generate() solver
    - For image generation tasks (text-to-image, image-gen, etc.): Uses custom image_generation_solver
    """
    print(
        f"Running reasoning benchmark with the following configuration:\n{OmegaConf.to_yaml(cfg)}"
    )

    dataset = load_dataset(cfg.dataset)

    if is_image_generation_task(cfg):
        # For image generation tasks we use custom solver and grader
        image_gen_solver = get_image_generation_solver(cfg.solver)
        grader = (
            get_image_grader(cfg.grader) if cfg.grader.get("enabled", True) else None
        )

        return Task(
            dataset=dataset,
            plan=[image_gen_solver],
            scorer=grader,
        )
    else:
        # Standard text generation plan
        model = get_solver(cfg.solver)
        grader = get_grader(cfg.grader) if cfg.grader.get("enabled", True) else None

        return Task(
            model=model,
            dataset=dataset,
            plan=[generate()],
            scorer=grader,
        )
