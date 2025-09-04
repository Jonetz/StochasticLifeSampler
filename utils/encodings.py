import json
import os
import re
import numpy as np
from typing import Optional, Sequence, List, Union, Tuple
import torch.nn.functional as F

import torch

_HEADER_RE = re.compile(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)", re.IGNORECASE)
_TOKENS_RE = re.compile(r"(\d*)([bo\$xyz!])", re.IGNORECASE)

# ---------------------------
# Utilities: RLE encode / decode
# ---------------------------
def rle_encode_binary(grid: torch.Tensor, rule: str = None) -> str:
    """
    Encode a 2D binary torch tensor into official Life RLE string.
    Header: "x = W, y = H" (plus ", rule = ..." if provided).
    Body: runs of alive (o) and dead (b), rows separated with $, terminated by !.
    """
    if isinstance(grid, np.ndarray):
        grid = torch.tensor(grid)
    if grid.ndim == 3 and grid.shape[0] in [0,1]:
        grid = grid.squeeze(0)
    assert grid.ndim == 2, "Expected 2D tensor"
    H, W = grid.shape
    parts = [f"x = {W}, y = {H}" + (f", rule = {rule}" if rule else ""), ]

    # Flatten row-major
    flat = grid.flatten().to(torch.uint8).cpu().numpy()
    runs = []
    cur = flat[0]
    cnt = 1
    for v in flat[1:]:
        v = int(v)
        if v == cur:
            cnt += 1
        else:
            char = 'o' if cur == 1 else 'b'
            runs.append(f"{cnt if cnt != 1 else ''}{char}")
            cur = v
            cnt = 1
    # last run
    char = 'o' if cur == 1 else 'b'
    runs.append(f"{cnt if cnt != 1 else ''}{char}")

    # Insert line breaks as needed, but simple implementation:
    body = "".join(runs) + "!"
    return "\n".join(parts) + "\n" + body

def rle_decode_binary(rle_str: str, device: Union[str, torch.device] = "cpu", target_shape: Tuple[int, int] = None) -> torch.Tensor:
    """
    Decode official Life RLE string into a torch tensor (H x W) binary.
    """
    lines = [ln.strip() for ln in rle_str.splitlines() if not ln.strip().startswith("#")]
    header = None
    for ln in lines:
        if _HEADER_RE.search(ln):
            header = ln
            break
    if header is None:
        raise ValueError("Missing RLE header (x = ..., y = ...)")
    m = _HEADER_RE.search(header)
    W, H = int(m.group(1)), int(m.group(2))

    # Body: everything after header line
    idx = lines.index(header)
    body = "".join(lines[idx + 1:])
    tensor = torch.zeros((H, W), dtype=torch.uint8, device=device)

    x = y = 0
    for count_str, tag in _TOKENS_RE.findall(body):
        count = int(count_str) if count_str else 1
        tag = tag.lower()
        if tag in ('b',):
            x += count
        elif tag in ('o', 'x', 'y', 'z'):  # treat multi-state as alive
            for i in range(count):
                if 0 <= y < H and 0 <= x < W:
                    tensor[y, x] = 1
                x += 1
        elif tag == '$':
            y += count
            x = 0
        elif tag == '!':
            break
        else:
            # unexpected tag, skip
            x += count
    
    # Optional: pad/cast to target_shape
    if target_shape is not None:
        H_max, W_max = target_shape
        if H_max < H or W_max < W:
            raise ValueError(f"Target shape {target_shape} is smaller than RLE array {(H,W)}")
        padded = np.zeros(target_shape, dtype=np.uint8)
        
        W, H = tensor.shape
        offset_w = (H_max - W) // 2
        offset_h = (W_max  - H) // 2

            # copy array into board
        padded[offset_h:offset_h+H, offset_w:offset_w+W] = tensor
        return padded


    return tensor

def save_rle_list(rle_list: Sequence[str], filepath: str):
    """
    Save a list of RLE strings to a file (JSON).
    Each entry corresponds to one chain's initial state.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for rle in rle_list:
            # Ensure actual newlines are written
            f.write(rle.strip() + "\n")



def load_rle_list(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        return json.load(f)

# ---------- ANALYSIS ----------
def load_rle_files(folder: str, max_files: int = None, target_shape=(200,200)):
    arrs = []
    sizes = []
    alives = []
    names = []
    files = [f for f in os.listdir(folder) if f.endswith(".rle")]
    if max_files:
        files = files[:max_files]
    skipped_files = 0
    print(f'Loading files ...')
    for fn in files:
        path = os.path.join(folder, fn)
        with open(path, "r") as f:
            txt = f.read()
        try:
            arr = rle_decode_binary(txt, target_shape=target_shape)# parse_rle(txt, target_shape=target_shape)
            arrs.append(arr)
            sizes.append(arr.shape[0] * arr.shape[1])
            alives.append(int(arr.sum()))
            names.append(fn)
        except Exception as e:
            skipped_files += 1
    print(f'Loaded in total {len(arrs)}. Skipped {skipped_files} files during loading (either failed to parse or the grid is too large)')
    return arrs, sizes, alives, names

@torch.no_grad()
def board_hash(states: torch.Tensor, K: torch.Tensor, rotate: bool = True) -> torch.Tensor:
    """
    Compute a translational (and optionally rotational) equivariant hash for a batch of 2D boards.

    Args:
        states: (B, H, W) tensor, dtype=bool or uint8
        K: (kH, kW) integer tensor for hashing. Must be precomputed and fixed for all comparisons.
        rotate: if True, compute hashes for 4 rotations (0, 90, 180, 270 degrees) and pick max

    Returns:
        hashes: (B,) int64 tensor, unique-ish per board up to translation and optionally rotation.
    """
    if states.dim() == 2:
        states = states.unsqueeze(0)
    elif states.dim() != 3:
        raise ValueError(f'The states tensor must have 2 or 3 dimensions not {states.dim()} and shape {states.shape}')
    B, H, W = states.shape
    kH, kW = K.shape

    # Convert to int64 for deterministic hash computation
    s = states#.to(torch.int64)

    # Add batch and channel dimensions for conv2d
    s = s.unsqueeze(1)  # (B,1,H,W)
    K = K.unsqueeze(0).unsqueeze(0)  # (1,1,kH,kW)

    # Convolution to compute translational hash
    conv_out = F.conv2d(s, K, padding=kH//2)  # (B,1,H,W)
    conv_out = conv_out.squeeze(1)  # (B,H,W)

    if rotate:
        convs = [conv_out]
        for _ in range(3):
            conv_out = torch.rot90(conv_out, 1, dims=(1,2))  # rotate 90 deg
            convs.append(conv_out)
        # Take the max over rotations to get rotation-equivariant hash
        conv_out = torch.stack(convs, dim=0).amax(dim=0)  # (B,H,W)

    # Collapse H,W into single number per board
    powers = torch.arange(1, H*W + 1, device=states.device, dtype=torch.int64)
    flat = conv_out.view(B, -1)
    hashes = (flat * powers).sum(dim=1)

    return hashes