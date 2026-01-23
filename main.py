import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from inspect_ai import eval
from omegaconf import DictConfig, OmegaConf

from reasoning_blind_spots.task import reasoning_benchmark as rb_task

load_dotenv()

OmegaConf.register_new_resolver("clean_model_name", lambda s: s.split("/")[-1])


@hydra.main(config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Wrapper task to run the reasoning benchmark from the root directory.
    """
    # Save Inspect AI logs in the same directory as Hydra outputs
    log_dir = HydraConfig.get().runtime.output_dir

    # Optional parameters for Inspect AI eval
    max_connections = cfg.get("max_connections", 15)

    # Eval resilience and limits
    time_limit = cfg.get("time_limit", cfg.get("timeout", None))
    fail_on_error = cfg.get("fail_on_error", True)
    retry_on_error = cfg.get("retry_on_error", 0)

    eval(
        rb_task(cfg),
        log_dir=log_dir,
        max_connections=max_connections,
        time_limit=time_limit,
        fail_on_error=fail_on_error,
        retry_on_error=retry_on_error,
    )


if __name__ == "__main__":
    run()
