from copy import deepcopy

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, prompt_template, solver
from inspect_ai.util import message_limit
from omegaconf import DictConfig, OmegaConf

SOLVER_PROMPT_TEMPLATE = """
Answer the user's question as accurately as possible.
State your final answer explicitly and clearly.

Question:
{prompt}
"""

TOOL_ENABLED_SOLVER_PROMPT_TEMPLATE = """
Answer the user's question as accurately as possible.
Use tools only when they are necessary to compute or verify the answer.
State your final answer explicitly and clearly.

Question:
{prompt}
"""


def _to_python(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return deepcopy(value)


def get_solver(
    solver_config: DictConfig = None,
) -> Model:
    """
    Returns the model
    """

    model_id = solver_config.get("model_id")
    has_backend_model = "backend" in solver_config and "model_name" in solver_config
    if model_id is None and not has_backend_model:
        raise ValueError(
            "solver_config must contain either 'model_id' or both 'backend' and 'model_name'."
        )

    model_str = model_id or (solver_config.backend + "/" + solver_config.model_name)
    gen_config = _to_python(solver_config.get("generate_config", {})) or {}

    # Any other argument in the config is passed to the model
    model_args = {
        k: _to_python(v)
        for k, v in solver_config.items()
        if k not in ["backend", "model_name", "model_id", "generate_config", "tools"]
    }

    if "gemma-4" in model_str:
        extra_body = dict(gen_config.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", True)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        gen_config["extra_body"] = extra_body

    print(
        f"Initializing solver model '{model_str}' with config: {gen_config} and args: {model_args}"
    )

    return get_model(
        model=model_str,
        role="solver",
        config=GenerateConfig(**gen_config),
        **model_args,
    )


def get_text_solver_prompt(tool_enabled: bool = False) -> Solver:
    """
    Returns a minimal prompt prefix for text solvers.
    """

    template = (
        TOOL_ENABLED_SOLVER_PROMPT_TEMPLATE if tool_enabled else SOLVER_PROMPT_TEMPLATE
    )
    return prompt_template(template)


@solver
def generate_with_tool_loop(max_additional_messages: int = 5) -> Solver:
    """
    Generate with tool calls enabled, while capping the number of
    additional conversation messages the solver may consume.
    """

    if max_additional_messages < 1:
        raise ValueError("max_additional_messages must be at least 1.")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        limit = len(state.messages) + max_additional_messages

        # Respect any task-level message cap if one is already in place.
        if state.message_limit is not None:
            limit = min(limit, state.message_limit)

        with message_limit(limit):
            return await generate(state, tool_calls="loop")

    return solve
