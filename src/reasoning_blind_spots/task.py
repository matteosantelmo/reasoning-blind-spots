from inspect_ai import Task, task
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import code_execution, web_search
from omegaconf import DictConfig, OmegaConf

from reasoning_blind_spots.dataset import load_dataset
from reasoning_blind_spots.grader import get_grader
from reasoning_blind_spots.image_grader import get_image_grader
from reasoning_blind_spots.image_solver import get_image_generation_solver
from reasoning_blind_spots.solver import (
    generate_with_tool_loop,
    get_solver,
    get_text_solver_prompt,
)

_INTERNAL_WEB_SEARCH_PROVIDERS = {
    "anthropic": "anthropic",
    "google": "gemini",
    "grok": "grok",
    "mistral": "mistral",
    "openai": "openai",
    "perplexity": "perplexity",
}


def is_image_generation_task(cfg: DictConfig) -> bool:
    if "question_type" not in cfg.dataset:
        return False

    question_types = set(cfg.dataset.question_type)
    return bool(
        question_types
        & {"text-to-image", "image-gen", "multi-to-image", "image-to-image"}
    )


def _to_python(value):
    """Convert OmegaConf containers to plain Python objects when needed."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _solver_provider(cfg: DictConfig) -> str | None:
    model_id = cfg.solver.get("model_id")
    if model_id:
        return str(model_id).split("/", 1)[0]
    return cfg.solver.get("backend")


def _resolve_web_search_providers(cfg: DictConfig, providers):
    resolved = _to_python(providers)
    if resolved is not None:
        return resolved

    provider = _solver_provider(cfg)
    if provider in _INTERNAL_WEB_SEARCH_PROVIDERS:
        return _INTERNAL_WEB_SEARCH_PROVIDERS[provider]

    raise ValueError(
        "solver.tools.web_search requires explicit "
        "solver.tools.web_search_providers for self-hosted or "
        "OpenAI-compatible models (for example ['tavily'])."
    )


def get_text_solver_plan(cfg: DictConfig):
    solver_tools = cfg.solver.get("tools", {})
    tool_enabled = solver_tools.get("enabled", False)
    plan = [get_text_solver_prompt(tool_enabled=tool_enabled)]

    if not tool_enabled:
        plan.append(generate())
        return plan

    tools = []
    if solver_tools.get("code_execution", True):
        providers = _to_python(solver_tools.get("providers", None))
        tools.append(code_execution(providers=providers))

    if solver_tools.get("web_search", False):
        providers = _resolve_web_search_providers(
            cfg, solver_tools.get("web_search_providers", None)
        )
        tools.append(web_search(providers=providers))

    if not tools:
        plan.append(generate())
        return plan

    plan.extend(
        [
            use_tools(
                tools,
                tool_choice=solver_tools.get("tool_choice", "auto"),
            ),
            generate_with_tool_loop(
                max_additional_messages=solver_tools.get("max_additional_messages", 5)
            ),
        ]
    )
    return plan


def get_task_sandbox(cfg: DictConfig):
    sandbox = _to_python(cfg.get("sandbox", None))
    if isinstance(sandbox, list):
        return tuple(sandbox)
    return sandbox


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
        task_sandbox = get_task_sandbox(cfg)

        task_kwargs = {}
        if task_sandbox is not None:
            task_kwargs["sandbox"] = task_sandbox

        return Task(
            model=model,
            dataset=dataset,
            solver=get_text_solver_plan(cfg),
            scorer=grader,
            **task_kwargs,
        )
