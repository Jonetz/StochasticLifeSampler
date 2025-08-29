import json
import numpy as np
from typing import Optional, Sequence, List

import torch
# ---------------------------
# Utilities: RLE encode / decode
# ---------------------------
def rle_encode_binary(grid: np.ndarray) -> str:
    """
    Encode a 2D binary numpy array into a simple run-length string.
    Format: "H W;val run,val run,..." where runs go row-major.
    Example: "5 5;0 10,1 3,0 12,..."
    This is compact and easy to parse; not the canonical .rle Life format,
    but simple and reversable.
    """
    assert grid.ndim == 2
    H, W = grid.shape
    flat = grid.reshape(-1).astype(np.uint8)
    runs = []
    if flat.size == 0:
        return f"{H} {W};"
    cur = int(flat[0])
    cnt = 1
    for v in flat[1:]:
        v = int(v)
        if v == cur:
            cnt += 1
        else:
            runs.append(f"{cur} {cnt}")
            cur = v
            cnt = 1
    runs.append(f"{cur} {cnt}")
    return f"{H} {W};" + ",".join(runs)


def rle_decode_binary(rle_str: str) -> np.ndarray:
    """
    Decode string from rle_encode_binary.
    """
    header, *rest = rle_str.split(";")
    H, W = map(int, header.split())
    if len(rest) == 0 or rest[0] == "":
        return np.zeros((H, W), dtype=np.uint8)
    runs = rest[0].split(",")
    flat = []
    for r in runs:
        val, cnt = r.strip().split()
        flat.extend([int(val)] * int(cnt))
    arr = np.array(flat, dtype=np.uint8)
    assert arr.size == H * W, f"Expected {H*W} elements, got {arr.size}"
    return arr.reshape((H, W))


def save_rle_list(rle_list: Sequence[str], filepath: str):
    """
    Save a list of RLE strings to a file (JSON).
    Each entry corresponds to one chain's initial state.
    """
    with open(filepath, "w") as f:
        json.dump(list(rle_list), f)


def load_rle_list(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        return json.load(f)

@torch.no_grad()
def board_hash(states: torch.Tensor, powers: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute per-board hash for a batch of boards.

    Args:
        states: (N,H,W) bool or uint8 tensor
        powers: (H*W,) precomputed tensor for hashing. If None, generates on the fly.

    Returns:
        hashes: (N,) int64 tensor
    """
    N, H, W = states.shape
    flat = states.view(N, -1).to(torch.int64)
    if powers is None:
        powers = torch.arange(1, H*W + 1, device=states.device, dtype=torch.int64)
    return (flat * powers).sum(dim=1)
