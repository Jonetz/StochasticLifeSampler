
from typing import Optional, Tuple, List, Union, Sequence, Dict
import numpy as np
try:
    import torch
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
        if states.ndim == 2:
            states = states.unsqueeze(0)
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
        """
        Return a deep copy of the Board.

        Returns:
            Board: a new Board object with copied tensor and metadata.

        Safeguards:
            - Copies the internal tensor and metadata to avoid side effects.
        """
        return Board(self._states.clone(), self.idx, device=self.device, meta=self.meta.copy())

    def to(self, device: Union[str, torch.device]) -> "Board":
        """
        Move the board tensor to a different device.

        Args:
            device: target torch device (str or torch.device)

        Returns:
            Board: new Board object on the specified device

        Safeguards:
            - Clones metadata to prevent accidental sharing.
        """
        return Board(self._states.to(device), self.idx, device=device, meta=self.meta.copy())

    def get_numpy(self, index: int = 0) -> np.ndarray:
        """
        Get a CPU numpy array for a single chain.

        Args:
            index: index of chain to extract (default 0)

        Returns:
            np.ndarray: array of shape (H, W), dtype uint8

        Safeguards:
            - Returns detached tensor to avoid modifying board.
            - Ensures index is valid.
        """
        t = self._states[index] if self.idx is None else self._states[self.idx]
        return t.detach().cpu().numpy().astype(np.uint8)

    def set_state(self, state: Union[np.ndarray, torch.Tensor], index: int = 0):
        """
        Replace the state of a chain with a new array.

        Args:
            state: 2D array or tensor of shape (H, W)
            index: index of chain to replace (default 0)

        Safeguards:
            - Checks shape compatibility with board dimensions.
            - Moves tensor to the board device.
        """
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state.astype(np.uint8)).to(self._states.device)
        else:
            state_t = state.to(self._states.device)
        assert state_t.shape == (self.H(), self.W()), f"Shape mismatch, got {state_t.shape}"
        self._states[index] = state_t
        
    @classmethod
    def from_shape(cls, 
                N: int, H: int, W: int, 
                device: Union[str, torch.device] = "cuda", 
                fill_prob: float = 0.0,
                fill_shape: Optional[Tuple[int, int]] = None):
        """
        Create a batched board of shape (N, H, W) with optional random initialization.

        Args:
            N: number of boards in batch
            H, W: board dimensions
            device: torch device
            fill_prob: probability of a cell being alive
            fill_shape: optional (H_box, W_box) subregion to fill; centered if provided

        Returns:
            Board: batched board object

        Safeguards:
            - Checks that fill_shape does not exceed board size.
            - Ensures tensor dtype is uint8.
        """
        t = torch.zeros((N, H, W), dtype=torch.uint8, device=device)

        if fill_prob > 0.0:
            if fill_shape is None:
                # fill entire board
                t = (torch.rand((N, H, W), device=device) < fill_prob).to(torch.uint8)
            else:
                Ah, Aw = fill_shape
                if Ah > H or Aw > W:
                    raise ValueError(f"fill_shape {fill_shape} cannot be larger than board size {(H, W)}")
                
                # compute offsets to center the subregion
                offset_h = (H - Ah) // 2
                offset_w = (W - Aw) // 2

                # fill only the subregion
                t[:, offset_h:offset_h+Ah, offset_w:offset_w+Aw] = (
                    (torch.rand((N, Ah, Aw), device=device) < fill_prob).to(torch.uint8)
                )

        return cls(t, idx=None, device=device, meta={})

    @classmethod
    def from_numpy_list(cls,
                        arrs: Sequence[np.ndarray],
                        device: Union[str, torch.device] = "cuda",
                        board_size: Optional[Tuple[int,int]] = None):
        """
        Build a batch of boards from a list of 2D numpy arrays.

        Args:
            arrs: sequence of 2D numpy arrays
            device: torch device
            board_size: optional target board size (H, W); arrays are centered if smaller

        Returns:
            Board: batched board object

        Safeguards:
            - Checks that each array fits within the target board size.
            - Converts arrays to uint8 and moves to correct device.
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
        Convert each board in the batch to its RLE string representation.

        Returns:
            List[str]: list of RLE strings, one per chain

        Safeguards:
            - Uses CPU numpy copy to avoid device conflicts.
            - Encodes each chain independently.
        """
        out = []
        for i in range(self.n_chains()):
            arr = self.get_numpy(i)
            out.append(rle_encode_binary(arr))
        return out

