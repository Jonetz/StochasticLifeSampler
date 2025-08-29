
from typing import Optional, Tuple, List, Union, Sequence, Dict
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("This module requires PyTorch (torch). Install PyTorch and retry.") from e

from utils.encodings import rle_encode_binary
# ---------------------------
# Board wrapper
# ---------------------------
class Board:
    """
    Lightweight wrapper around a batched states tensor.

    The backend tensor is shape (N, H, W) dtype torch.uint8 or bool.
    Board may represent the full batch (idx=None) or be a handle for a single index.
    """

    def __init__(self,
                 states: torch.Tensor,
                 idx: Optional[int] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 meta: Optional[Dict] = None):
        """
        states: (N, H, W) tensor
        idx: if provided, this Board is a view for chain index idx
        device: optional device to move tensor to
        meta: optional metadata dictionary per-board or per-batch
        """
        if device is not None:
            states = states.to(device)
        assert states.ndim == 3, "states must be a 3D tensor (N, H, W)"
        self._states = states
        self.idx = idx
        self.device = states.device
        self.meta = meta or {}

    @property
    def tensor(self) -> torch.Tensor:
        if self.idx is None:
            return self._states
        else:
            return self._states[self.idx].unsqueeze(0)  # shape (1,H,W)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self._states.shape)

    def n_chains(self) -> int:
        return self._states.shape[0]

    def H(self) -> int:
        return self._states.shape[1]

    def W(self) -> int:
        return self._states.shape[2]

    def clone(self) -> "Board":
        return Board(self._states.clone(), self.idx, device=self.device, meta=self.meta.copy())

    def to(self, device: Union[str, torch.device]) -> "Board":
        return Board(self._states.to(device), self.idx, device=device, meta=self.meta.copy())

    def get_numpy(self, index: int = 0) -> np.ndarray:
        """
        Return a CPU numpy array for chain index (default 0 if single).
        """
        t = self._states[index] if self.idx is None else self._states[self.idx]
        return t.detach().cpu().numpy().astype(np.uint8)

    def set_state(self, state: Union[np.ndarray, torch.Tensor], index: int = 0):
        """
        Replace the state of chain `index` with provided state array (H,W).
        """
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state.astype(np.uint8)).to(self._states.device)
        else:
            state_t = state.to(self._states.device)
        assert state_t.shape == (self.H(), self.W()), f"Shape mismatch, got {state_t.shape}"
        self._states[index] = state_t

    @classmethod
    def from_shape(cls, N: int, H: int, W: int, device: Union[str, torch.device] = "cuda", fill_prob: float = 0.0):
        """
        Create a Batched Board with random init (fill_prob) or empty.
        """
        if fill_prob <= 0.0:
            t = torch.zeros((N, H, W), dtype=torch.uint8, device=device)
        else:
            t = (torch.rand((N, H, W), device=device) < float(fill_prob)).to(torch.uint8)
        return cls(t, idx=None, device=device, meta={})

    @classmethod
    def from_numpy_list(cls,
                        arrs: Sequence[np.ndarray],
                        device: Union[str, torch.device] = "cuda",
                        board_size: Optional[Tuple[int,int]] = None):
        """
        Build batch from list of numpy 2D arrays.
        If board_size is provided and larger than array, the array is centered.
        """
        assert len(arrs) > 0
        # Determine max shape if board_size not given
        if board_size is None:
            H, W = arrs[0].shape
            board_size = (H, W)
        H_board, W_board = board_size
        N = len(arrs)

        t = torch.zeros((N, H_board, W_board), dtype=torch.uint8, device=device)

        for i, a in enumerate(arrs):
            H_a, W_a = a.shape
            if H_a > H_board or W_a > W_board:
                raise ValueError(f"Array {i} shape {a.shape} larger than target board {board_size}")

            # compute top-left offset to center
            offset_h = (H_board - H_a) // 2
            offset_w = (W_board - W_a) // 2

            # copy array into board
            t[i, offset_h:offset_h+H_a, offset_w:offset_w+W_a] = torch.from_numpy(a.astype(np.uint8)).to(device)
        return cls(t, idx=None, device=device)


    def rle_list(self) -> List[str]:
        """
        Return list of RLE strings for each chain (CPU numpy conversion).
        """
        out = []
        for i in range(self.n_chains()):
            arr = self.get_numpy(i)
            out.append(rle_encode_binary(arr))
        return out

