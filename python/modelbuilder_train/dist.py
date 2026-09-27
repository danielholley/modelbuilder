"""Data-parallel training under ``torchrun`` (one process per GPU, one or more machines).

Without ``torchrun`` everything here is a no-op, so single-process runs are unchanged.
Rank 0 writes the events and checkpoints; every rank samples its own windows.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def init(device: torch.device) -> torch.device:
    """Joins the process group when launched by torchrun; returns this rank's device."""
    if world() == 1 or dist.is_initialized():
        return device
    if device.type == "cuda":
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    return device


def wrap(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not dist.is_initialized():
        return module
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module, device_ids=[device.index] if device.type == "cuda" else None)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
