from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import message_limit


def get_solver(
    solver_config: dict = None,
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
    gen_config = solver_config.get("generate_config", {})

    # Any other argument in the config is passed to the model
    model_args = {
        k: v
        for k, v in solver_config.items()
        if k not in ["backend", "model_name", "model_id", "generate_config", "tools"]
    }

    return get_model(
        model=model_str,
        role="solver",
        config=GenerateConfig(**gen_config),
        **model_args,
    )


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
