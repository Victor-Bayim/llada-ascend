"""Device checks shared by maintained runners; never silently substitute CPU."""
from __future__ import annotations

import importlib
import re
import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if not re.fullmatch(r"npu(?::\d+)?", requested):
        raise ValueError("Use an explicit 'cpu' or 'npu:N' device.")
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("NPU requested but torch_npu is not installed. No CPU fallback.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU requested but no Ascend device is available. No CPU fallback.")
    torch.npu.set_device(requested)
    return torch.device(requested)


def configure_npu(device: torch.device, *, jit_compile: bool = False,
                  mm_bmm_nd: bool = False) -> None:
    if device.type != "npu":
        if jit_compile or mm_bmm_nd:
            raise ValueError("NPU tuning flags require --device npu:N.")
        return
    if jit_compile:
        torch.npu.set_compile_mode(jit_compile=True)
    if mm_bmm_nd:
        torch.npu.set_mm_bmm_format_nd(True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "npu":
        # Errors must propagate: reporting an unsynchronized NPU time is misleading.
        torch.npu.synchronize()


def seed_execution(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)
