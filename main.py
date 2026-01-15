import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from inspect_ai import eval
from omegaconf import DictConfig

from reasoning_blind_spots.task import reasoning_benchmark as rb_task

load_dotenv()


@hydra.main(config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Wrapper task to run the reasoning benchmark from the root directory.
    """
    # Save Inspect AI logs in the same directory as Hydra outputs
    log_dir = HydraConfig.get().runtime.output_dir

    # Optional parameters for Inspect AI eval
    max_connections = cfg.get("max_connections", 10)
    timeout = cfg.get("timeout", None)

    eval(
        rb_task(cfg), log_dir=log_dir, max_connections=max_connections, timeout=timeout
    )


if __name__ == "__main__":
    run()
