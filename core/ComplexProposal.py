from typing import Optional, Tuple
import torch
from core.Board import Board
from core.Proposal import Proposal
from utils.encodings import load_rle_files
from utils.neural_proposals import PatchNet

class PatternInsertProposal(Proposal):
    """
    Proposal that inserts known patterns (loaded from RLE files) into boards.
    - Loads all patterns at init (CPU), computes tight bounding boxes, pads to (max_h,max_w),
      and stores a GPU tensor patterns_padded (P, max_h, max_w).
    - propose(board): choose one pattern per batch entry and paste it in a random location
      inside the provided bounding box (self.box_size or full board).
    """

    def __init__(self,
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
        device: torch device to store patterns
        """
        super().__init__(name=name, box_size=box_size, device=device)
        self.rle_folder = rle_folder
        self.max_files = max_files
        self.target_shape = target_shape
        # Load patterns (numpy arrays) using your loader function
        arrs, names = load_rle_files(rle_folder, max_files=max_files, target_shape=target_shape or (200,200))
        if len(arrs) == 0:
            raise RuntimeError(f"No patterns loaded from {rle_folder}")

        # Convert arrays into tight-bbox patterns and compute sizes
        self.names = names
        patterns, sizes_h, sizes_w = [], [], []
        for a in arrs:
            if a is None or not a.any():
                continue

            # compute tight bounding box
            rows = a.any(dim=1)
            cols = a.any(dim=0)
            r_indices = torch.where(rows)[0]
            c_indices = torch.where(cols)[0]
            rmin, rmax = r_indices[0].item(), r_indices[-1].item()
            cmin, cmax = c_indices[0].item(), c_indices[-1].item()

            trimmed = a[rmin:rmax+1, cmin:cmax+1].to(torch.uint8)
            patterns.append(trimmed)
            sizes_h.append(trimmed.shape[0])
            sizes_w.append(trimmed.shape[1])

        if len(patterns) == 0:
            raise RuntimeError("No non-empty patterns found in folder")

        # max dimensions
        self.P = len(patterns)
        self.max_h = max(sizes_h)
        self.max_w = max(sizes_w)

        # padded tensor (P, max_h, max_w)
        pat_tensor = torch.zeros((self.P, self.max_h, self.max_w), dtype=torch.uint8, device=self.device)
        for idx, pat in enumerate(patterns):
            h, w = pat.shape
            pat_tensor[idx, :h, :w] = pat  # already tensor

        self.patterns_padded = pat_tensor

        # sizes as tensors on device
        self.sizes_h = torch.tensor(sizes_h, dtype=torch.long, device=self.device)
        self.sizes_w = torch.tensor(sizes_w, dtype=torch.long, device=self.device)

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
    
class AreaTransformProposal(Proposal):
    """
    Randomly selects a sub-area inside the bounding box and applies a random
    spatial transformation (flip or rotation).
    """
    def __init__(self, 
                 box_size: Optional[Tuple[int,int]] = None,
                 min_patch: int = 4,
                 max_patch: int = 20,
                 name: Optional[str] = None,):
        super().__init__(name, box_size=box_size)
        self.min_patch = min_patch
        self.max_patch = max_patch

    def propose(self, board: Board) -> Board:
        b = board.clone()
        N, H, W = b.shape

        # Define bounding box (defaults to full board)
        if self.box_size:
            Ah, Aw = self.box_size
            h_start = (H - Ah) // 2
            w_start = (W - Aw) // 2
            h_end = h_start + Ah
            w_end = w_start + Aw
        else:
            h_start, h_end = 0, H
            w_start, w_end = 0, W

        # Random board index
        idx = torch.randint(0, N, (1,)).item()

        # Choose random transform
        op = torch.randint(0, 5, (1,)).item()
        if op < 3:
            p = torch.randint(self.min_patch, self.max_patch + 1, (1,)).item()
            ph = pw = p
        else:
            # Random patch size
            ph = torch.randint(self.min_patch, self.max_patch + 1, (1,)).item()
            pw = torch.randint(self.min_patch, self.max_patch + 1, (1,)).item()

        # Random position (ensure patch fits inside bounding box)
        i = torch.randint(h_start, max(h_start+1, h_end - ph + 1), (1,)).item()
        j = torch.randint(w_start, max(w_start+1, w_end - pw + 1), (1,)).item()

        # Extract patch
        patch = b._states[idx, i:i+ph, j:j+pw]

        if op == 0:
            patch_new = torch.rot90(patch, 1, [0,1])   # 90°
        elif op == 1:
            patch_new = torch.rot90(patch, 2, [0,1])   # 180°
        elif op == 2:
            patch_new = torch.rot90(patch, 3, [0,1])   # 270°
        elif op == 3:
            patch_new = torch.flip(patch, [0])         # vertical flip
        elif op == 4:
            patch_new = torch.flip(patch, [1])         # horizontal flip

        # Write back
        b._states[idx, i:i+ph, j:j+pw] = patch_new

        return b

class PatchNetProposal(Proposal):
    """
    Uses trained model to a fill in a specific part of the patch net.
    """
    def __init__(self, 
                 filepath: str = f'data\network_final.pth',
                 box_size: Optional[Tuple[int,int]] = None,
                 env_size: int = 12,
                 patch_size: int = 3,
                 name: Optional[str] = None,
                 device: Optional[str] = None):
        super().__init__(name, box_size=box_size, device=device)
        self.env_size = env_size
        self.patch_size = patch_size
        weights = torch.load(filepath)
        self.network = PatchNet(env_size=env_size, patch_size=patch_size, dropout=0)
        self.network.load_state_dict(weights)
        self.network.eval()  # important for inference
        self.network.to(self.device)                
    
    def propose(self, board: Board) -> Board:
        b = board.clone()
        N, H, W = b.shape

        # Bounding box
        if self.box_size:
            Ah, Aw = self.box_size
            h_start = (H - Ah) // 2
            w_start = (W - Aw) // 2
            h_end = h_start + Ah - self.env_size
            w_end = w_start + Aw - self.env_size
        else:
            h_start, h_end = 0, H - self.env_size
            w_start, w_end = 0, W - self.env_size

        if self.env_size > (h_end - h_start) or self.env_size > (w_end - w_start):
            raise ValueError("Max pattern size larger than placement box.")

        # Random position inside bounding box
        i = torch.randint(h_start, h_end, (1,)).item()
        j = torch.randint(w_start, w_end, (1,)).item()

        # Extract environment patch
        env = b._states[:, i:i+self.env_size, j:j+self.env_size].to(self.device)
        env = env.unsqueeze(1).float()  # [N, 1, env_size, env_size]

        # Predict patch
        prop = self.network(env)  # [N, patch_size, patch_size]

        # Insert predicted patch into board (centered)
        center_i = self.env_size // 2 - self.patch_size // 2
        center_j = self.env_size // 2 - self.patch_size // 2
        h_insert = i + center_i
        w_insert = j + center_j

        b._states[:, h_insert:h_insert+self.patch_size, w_insert:w_insert+self.patch_size] = prop

        return b

class NewBoardProposal(Proposal):
    """
    Proposal operator that discards the current state and 
    proposes a completely new random board.

    Uses Board.from_shape for efficient batch initialization.

    Args:
        fill_prob: Probability of a cell being alive (default 0.35).
        fill_shape: Optional (H_box, W_box) subregion to fill; centered if provided.
        device: Torch device (default: inferred from CUDA availability).
    
    Methods:
        propose(board): Returns a new batch of boards with the same
                        batch size and dimensions as input, but filled randomly.

    Safeguards:
        - Preserves batch size N, height H, width W from input board.
        - Ensures returned board is on the correct device and dtype.
    """

    def __init__(self, 
                 fill_prob: float = 0.35, 
                 box_size: Optional[Tuple[int, int]] = None,
                 name: Optional[str] = None,
                 device: Optional[str] = None):
        super().__init__(name=name or "NewBoardProposal", box_size=None, device=device)
        self.fill_prob = fill_prob
        self.box_size = box_size

    def propose(self, board: Board) -> Board:
        """
        Discards the current state and proposes a new random board.

        Args:
            board (Board): Current board batch.

        Returns:
            Board: Newly sampled board batch with same dimensions as input.
        """
        N, H, W = board.tensor.shape

        # Re-sample completely new boards
        new_boards = Board.from_shape(
            N=N, H=H, W=W,
            device=self.device,
            fill_prob=self.fill_prob,
            fill_shape=self.box_size
        )
        return new_boards