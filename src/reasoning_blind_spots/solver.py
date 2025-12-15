from inspect_ai.model import GenerateConfig, Model, get_model


def get_solver(
    solver_config: dict = None,
) -> Model:
    """
    Returns the model
    """

    if "backend" not in solver_config or "model_name" not in solver_config:
        raise ValueError("solver_config must contain 'backend' and 'model_name' keys.")

    if solver_config.backend == "vllm" and "model_base_url" not in solver_config:
        raise ValueError(
            "For 'vllm' backend, 'model_base_url' must be specified in solver_config."
        )

    model_str = solver_config.backend + "/" + solver_config.model_name
    model_url = solver_config.get("model_base_url", None)
    api_key = solver_config.get("api_key", None)
    gen_config = solver_config.get("generate_config", {})

    return get_model(
        model=model_str,
        role="solver",
        base_url=model_url,
        api_key=api_key,
        config=GenerateConfig(**gen_config),
    )
