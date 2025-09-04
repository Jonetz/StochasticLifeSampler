from typing import Optional, Tuple
import numpy as np
import torch
from core import Board
from core.Proposal import Proposal
from utils.encodings import load_rle_files

# Pattern Proposal

class PatternInsertProposal(Proposal):
    """
    Proposal that inserts known patterns (loaded from RLE files) into boards.
    - Loads all patterns at init (CPU), computes tight bounding boxes, pads to (max_h,max_w),
      and stores a GPU tensor patterns_padded (P, max_h, max_w).
    - propose(board): choose one pattern per batch entry and paste it in a random location
      inside the provided bounding box (self.box_size or full board).
    """

    def __init__(self,
                 engine,
                 rle_folder: str,
                 max_files: Optional[int] = None,
                 name: Optional[str] = None,
                 box_size: Optional[Tuple[int,int]] = None,
                 target_shape: Optional[Tuple[int,int]] = None,
                 device: Optional[torch.device] = None):
        """
        rle_folder: path to folder with .rle files
        max_files: limit number of patterns to load
        box_size: optional (H_box, W_box) for placement; if None uses full board during propose
        target_shape: optional target shape used when decoding RLEs (if loader supports it)
        device: torch device to store patterns (defaults to engine.device)
        """
        super().__init__(engine, name=name, box_size=box_size)
        self.rle_folder = rle_folder
        self.max_files = max_files
        self.device = torch.device(device) if device is not None else engine.device
        self.target_shape = target_shape
        # Load patterns (numpy arrays) using your loader function
        arrs, _, _, names = load_rle_files(rle_folder, max_files=max_files, target_shape=target_shape or (200,200))
        if len(arrs) == 0:
            raise RuntimeError(f"No patterns loaded from {rle_folder}")

        # Convert arrays into tight-bbox patterns and compute sizes
        self.names = names
        patterns = []   # list of small numpy masks
        sizes = []
        for a in arrs:
            # a is 2D numpy binary array
            if not np.any(a):
                # skip empty
                continue
            rows = np.any(a, axis=1)
            cols = np.any(a, axis=0)
            rmin, rmax = np.where(rows)[0][0], np.where(rows)[0][-1]
            cmin, cmax = np.where(cols)[0][0], np.where(cols)[0][-1]
            small = a[rmin:rmax+1, cmin:cmax+1].astype(np.uint8)
            patterns.append(small)
            sizes.append(small.shape)

        if len(patterns) == 0:
            raise RuntimeError("No non-empty patterns found in folder")

        # compute max patch size
        self.P = len(patterns)
        self.sizes = sizes  # list of (h,w)
        self.max_h = max(s[0] for s in sizes)
        self.max_w = max(s[1] for s in sizes)

        # create padded patterns tensor (P, max_h, max_w) on device
        pat_tensor = torch.zeros((self.P, self.max_h, self.max_w), dtype=torch.uint8)
        for idx, pat in enumerate(patterns):
            h, w = pat.shape
            pat_tensor[idx, :h, :w] = torch.from_numpy(pat)

        self.patterns_padded = pat_tensor.to(self.device)  # uint8 (P, max_h, max_w)
        # Also store sizes as tensors for quick indexing on device
        hs = torch.tensor([s[0] for s in sizes], dtype=torch.long, device=self.device)
        ws = torch.tensor([s[1] for s in sizes], dtype=torch.long, device=self.device)
        self.sizes_h = hs
        self.sizes_w = ws

    def propose(self, board: Board) -> Board:
        """
        Insert one chosen pattern per batch entry into the board at a random top-left position
        inside the configured box (or whole board).
        Fully vectorized.
        """
        b = board.clone()
        N, H, W = b.shape
        device = b._states.device

        # Determine bounding-box for placement (use self.box_size if not None, else full board)
        if self.box_size is not None:
            box_H, box_W = self.box_size
            if box_H > H or box_W > W:
                raise ValueError("box_size larger than board")
            start_i = (H - box_H) // 2
            start_j = (W - box_W) // 2
            end_i = start_i + box_H
            end_j = start_j + box_W
        else:
            start_i, start_j, end_i, end_j = 0, 0, H, W
            box_H = H
            box_W = W

        # If pattern max size doesn't fit into the box, we cannot place
        if self.max_h > (end_i - start_i) or self.max_w > (end_j - start_j):
            raise ValueError("Max pattern size larger than placement box. Increase box_size or filter patterns.")

        # 1) sample pattern index per batch
        # Uniform random among self.P patterns
        p_idx = torch.randint(0, self.P, (N,), device=device, dtype=torch.long)  # (N,)

        # 2) sample top-left positions per batch such that the full padded (max_h,max_w) block fits
        max_placerows = (end_i - start_i) - self.max_h + 1  # >=1
        max_placecols = (end_j - start_j) - self.max_w + 1
        # sample offsets in [0, max_placerows-1], then add start_i
        offs_i = torch.randint(0, max_placerows, (N,), device=device, dtype=torch.long) + start_i
        offs_j = torch.randint(0, max_placecols, (N,), device=device, dtype=torch.long) + start_j
        # Note: because each pattern may be smaller than max_h/max_w, padded area contains zeros and will be pasted safely.

        # 3) gather chosen patterns (N, max_h, max_w)
        chosen = self.patterns_padded[p_idx]  # (N, max_h, max_w) uint8

        # 4) Build index grids to place patches into board tensor
        # rows: shape (N, max_h) -> offs_i[:,None] + arange(max_h)
        rows = offs_i.unsqueeze(1) + torch.arange(self.max_h, device=device).unsqueeze(0)  # (N, max_h)
        cols = offs_j.unsqueeze(1) + torch.arange(self.max_w, device=device).unsqueeze(0)  # (N, max_w)
        # Expand to (N, max_h, max_w) for broadcasted assignment
        rows_idx = rows.unsqueeze(2).expand(-1, -1, self.max_w)   # (N, max_h, max_w)
        cols_idx = cols.unsqueeze(1).expand(-1, self.max_h, -1)   # (N, max_h, max_w)

        batch_idx = torch.arange(N, device=device, dtype=torch.long).view(N, 1, 1).expand(-1, self.max_h, self.max_w)

        # 5) Paste: chosen is uint8 pattern mask; we can OR it into board or replace the region.
        # Here we choose to XOR (flip) or OR? Typically insertion should set alive cells (1),
        # but user wanted "place the loaded figure somewhere in the field" — we'll set those cells to 1.
        b._states[batch_idx, rows_idx, cols_idx] = torch.where(
            chosen.to(b._states.dtype) > 0,
            torch.ones_like(chosen, dtype=b._states.dtype),
            b._states[batch_idx, rows_idx, cols_idx]
        )

        return b
    
# Neural Network proposal
