"""
Launcher for all experiments
"""

import os
import sys
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # only if you’re sure this runs before any CUDA use

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _maybe_limit_cuda_memory():
    memory_limit_gb = os.environ.get("DICE_RL_CUDA_MEMORY_LIMIT_GB")
    if memory_limit_gb is None:
        memory_limit_gb = os.environ.get("PYTORCH_CUDA_MEMORY_LIMIT_GB")
    if not memory_limit_gb:
        return

    try:
        memory_limit_gb = float(memory_limit_gb)
    except ValueError as exc:
        raise ValueError(
            "DICE_RL_CUDA_MEMORY_LIMIT_GB must be a number of GiB, "
            f"got {memory_limit_gb!r}"
        ) from exc
    if memory_limit_gb <= 0:
        raise ValueError("DICE_RL_CUDA_MEMORY_LIMIT_GB must be positive")

    import torch

    if not torch.cuda.is_available():
        log.warning("CUDA memory limit requested, but CUDA is not available")
        return

    limit_bytes = memory_limit_gb * 1024**3
    for device_idx in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        fraction = min(limit_bytes / total_bytes, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)
        log.info(
            "Set CUDA memory cap on visible device %s to %.2f GiB "
            "(%.1f%% of %.2f GiB)",
            device_idx,
            min(memory_limit_gb, total_bytes / 1024**3),
            fraction * 100,
            total_bytes / 1024**3,
        )


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    # NOTE: Don't resolve the entire config here! This breaks configs where
    # agents need to update values before interpolations are resolved.
    # OmegaConf.resolve(cfg)
    _maybe_limit_cuda_memory()
    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
