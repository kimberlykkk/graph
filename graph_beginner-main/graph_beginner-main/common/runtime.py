from __future__ import annotations

import json
import random
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch


def configure_cpu(seed: int, threads: int) -> torch.device:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    return torch.device("cpu")


def save_result(result: dict[str, Any], output: str | None) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def make_neighbor_loader(*args: Any, **kwargs: Any):
    try:
        from torch_geometric.loader import NeighborLoader
    except ImportError as exc:
        raise RuntimeError("NeighborLoader is not available in this PyG installation.") from exc
    if not any(find_spec(name) is not None for name in ("pyg_lib", "torch_sparse")):
        raise RuntimeError(
            "Sampled mode requires pyg-lib or torch-sparse. "
            "Install a wheel matching your Python, PyTorch and CPU versions; "
            "full-graph mode does not need this optional extension."
        )
    try:
        return NeighborLoader(*args, **kwargs)
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "PyG NeighborLoader requires a compatible CPU pyg-lib or torch-sparse installation. "
            "Install the matching extension from https://data.pyg.org/whl/."
        ) from exc
