from inspect_ai import Task, task
from inspect_ai.solver import generate
from omegaconf import DictConfig, OmegaConf

from reasoning_blind_spots.dataset import load_dataset
from reasoning_blind_spots.grader import get_grader
from reasoning_blind_spots.solver import get_solver


@task
def reasoning_benchmark(cfg: DictConfig):
    """
    Defines the reasoning benchmark task.
    """
    print(
        f"Running reasoning benchmark with the following configuration:\n{OmegaConf.to_yaml(cfg)}"
    )

    dataset = load_dataset(cfg.dataset)
    solver = get_solver(cfg.solver)
    grader = get_grader(cfg.grader)

    return Task(model=solver, dataset=dataset, plan=[generate()], scorer=grader)
