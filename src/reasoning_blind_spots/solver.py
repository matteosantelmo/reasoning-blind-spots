from inspect_ai.model import GenerateConfig, Model, get_model


def get_solver(
    solver_config: dict = None,
) -> Model:
    """
    Returns the model
    """

    if "backend" not in solver_config or "model_name" not in solver_config:
        raise ValueError("solver_config must contain 'backend' and 'model_name' keys.")

    model_str = solver_config.backend + "/" + solver_config.model_name
    gen_config = solver_config.get("generate_config", {})

    # Any other argument in the config is passed to the model
    model_args = {
        k: v
        for k, v in solver_config.items()
        if k not in ["backend", "model_name", "generate_config"]
    }

    return get_model(
        model=model_str,
        role="solver",
        config=GenerateConfig(**gen_config),
        **model_args,
    )
