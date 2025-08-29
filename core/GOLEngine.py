
import os
import sys
# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, Tuple, List, Union, Sequence
import numpy as np
from core.Board import Board
from utils.encodings import save_rle_list, load_rle_list, rle_decode_binary
from PIL import Image

"""
gol_engine.py

Batched GPU-accelerated Game of Life engine + Board wrapper.

Requirements:
    - torch (PyTorch) >=1.8
    - numpy
    - imageio
    - matplotlib

Focus: clear OOP design, batched tensor backend, RLE save/load, gif visualization.
"""


try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("This module requires PyTorch (torch). Install PyTorch and retry.") from e

try:
    import imageio
    import matplotlib.pyplot as plt
except Exception:
    # Visualization will raise helpful error if missing when called
    imageio = None
    plt = None

# ---------------------------
# GoLEngine
# ---------------------------
class GoLEngine:
    """
    Batched GPU-accelerated Game of Life engine.

    Usage:
        engine = GoLEngine(device='cuda', border='wrap')
        board = Board.from_shape(N=64, H=50, W=50, device='cuda', fill_prob=0.1)
        final = engine.simulate(board, steps=100, return_trajectory=False)
    """

    NEIGH_KERNEL = torch.tensor([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=torch.float32)    
    def __init__(self,
                 device: Union[str, torch.device] = "cuda",
                 border: str = "wrap",
                 skip_osci: bool = True):
        self.device = torch.device(device)
        self.border = border
        self.skip_osci = skip_osci
        self.kernel = GoLEngine.NEIGH_KERNEL.view(1, 1, 3, 3).to(self.device)
        # map string -> int flag for JIT
        self.pad_flag = {"wrap": 0, "constant": 1, "reflect": 2}[border]

    def _pad_mode_and_kwargs(self):
        if self.border == "wrap":
            return "circular", {}
        elif self.border == "constant":
            return "constant", {"value": 0}
        elif self.border == "reflect":
            return "reflect", {}
        else:
            raise ValueError(f"Unknown border mode {self.border}")

    @torch.jit.script
    def step_jit(states: torch.Tensor, kernel: torch.Tensor, pad_flag: int) -> torch.Tensor:
        x = states.float().unsqueeze(1)
        if pad_flag == 0:
            x_p = F.pad(x, (1, 1, 1, 1), mode='circular')
        elif pad_flag == 1:
            x_p = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
        else:
            x_p = F.pad(x, (1, 1, 1, 1), mode='reflect')
        nbh = F.conv2d(x_p, kernel).squeeze(1)
        survive = states & ((nbh == 2) | (nbh == 3))
        born = (~states) & (nbh == 3)
        return survive | born

    @torch.no_grad()
    def simulate(self,
                 board: Union['Board', torch.Tensor],
                 steps: int = 1,
                 return_trajectory: bool = False,
                 progress_callback: Optional[callable] = None
                 ) -> Union['Board', Tuple['Board', List[torch.Tensor]]]:

        if isinstance(board, Board):
            states = board._states.bool()
        else:
            states = board.bool()
        N, H, W = states.shape

        traj: Optional[List[torch.Tensor]] = [] if return_trajectory else None
        seen_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
        s = states
        if return_trajectory:
            traj.append(s.clone())

        # Precompute powers for hashing
        flat_size = H * W
        powers = torch.arange(1, flat_size + 1, device=self.device, dtype=torch.int64)

        for t in range(steps):
            s = GoLEngine.step_jit(s, self.kernel, self.pad_flag)

            if progress_callback:
                progress_callback(t + 1, s)

            if return_trajectory:
                traj.append(s.clone())
            elif self.skip_osci:
                flat = s.view(N, -1).to(torch.int64)
                hashes = (flat * powers).sum(dim=1)
                mask = torch.isin(hashes, seen_hashes)
                if mask.any():
                    # Oscillation detected
                    final_board = Board(s, idx=None,
                                        device=self.device,
                                        meta=(board.meta.copy() if isinstance(board, Board) else {}))
                    return final_board
                seen_hashes = torch.cat([seen_hashes, hashes])

        final_board = Board(s, idx=None,
                            device=self.device,
                            meta=(board.meta.copy() if isinstance(board, Board) else {}))
        if return_trajectory:
            return final_board, traj
        else:
            return final_board

    # ---------------------------
    # Convenience: wrappers to set initial states
    # ---------------------------
    def make_random(self, N: int, H: int, W: int, fill_prob: float = 0.1) -> Board:
        return Board.from_shape(N=N, H=H, W=W, device=self.device, fill_prob=fill_prob)

    def from_numpy_list(self, arrs: Sequence[np.ndarray]) -> Board:
        return Board.from_numpy_list(arrs, device=self.device)

    # ---------------------------
    # Visualization and saving
    # ---------------------------    
    def trajectory_to_gif(self,
                        trajectory: Sequence[torch.Tensor],
                        filepath: str,
                        fps: int = 10,
                        scale: int = 4,
                        invert: bool = True):
        """
        Save a trajectory to GIF. Corrected for viewing issues.
        """
        frames = []
        for t in trajectory:
            # convert to numpy 2D array
            if isinstance(t, torch.Tensor):
                if t.ndim == 3:
                    arr = t[0].detach().cpu().numpy()
                elif t.ndim == 2:
                    arr = t.detach().cpu().numpy()
                else:
                    raise ValueError("Trajectory tensors must be 2D or 3D")
            else:
                arr = np.asarray(t)

            if invert:
                arr = 1 - arr
            # scale up for visibility
            if scale != 1:
                arr = np.kron(arr, np.ones((scale, scale)))

            # convert to uint8 0-255
            arr_uint8 = (arr * 255).astype(np.uint8)
            # create PIL Image in 'L' mode (grayscale)
            img = Image.fromarray(arr_uint8, mode='L')
            frames.append(img)

        # duration per frame in ms
        duration_ms = int(1000 / fps)
        frames[0].save(filepath, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)

    def save_initial_states_rle(self, board: Board, filepath: str):
        rles = board.rle_list()
        save_rle_list(rles, filepath)

    def load_initial_states_rle(self, filepath: str) -> Board:
        rles = load_rle_list(filepath)
        arrs = [rle_decode_binary(s) for s in rles]
        return self.from_numpy_list(arrs)

    def show(self, idx: int = 0, cmap: str = "gray", invert: bool = True, scale: int = 4):
        """
        Show a single configuration (e.g. the initial state).
        idx: index of chain (ignored if only one board)
        """
        if plt is None:
            raise ImportError("matplotlib is required for visualization.")
        state = self.state[idx].detach().cpu().numpy() if self.state.ndim == 3 else self.state.detach().cpu().numpy()
        if invert:
            state = 1 - state
        if scale != 1:
            state = np.kron(state, np.ones((scale, scale)))
        plt.imshow(state, cmap=cmap)
        plt.axis("off")
        plt.show()

    def show_trajectory(self,
                    trajectory: Sequence[torch.Tensor],
                    cmap: str = "gray",
                    scale: int = 4,
                    invert: bool = True,
                    pause: float = 0.5):
        """
        Display a trajectory frame by frame using matplotlib.
        """
        if plt is None:
            raise ImportError("matplotlib is required for visualization.")

        for i, t in enumerate(trajectory):
            if isinstance(t, torch.Tensor):
                if t.ndim == 3:
                    arr = t[0].detach().cpu().numpy()
                elif t.ndim == 2:
                    arr = t.detach().cpu().numpy()
                else:
                    raise ValueError("Trajectory tensors must be 2D or 3D")
            else:
                arr = np.asarray(t)

            if invert:
                arr = 1 - arr
            if scale != 1:
                arr = np.kron(arr, np.ones((scale, scale)))

            plt.imshow(arr, cmap=cmap)
            plt.title(f"Frame {i}")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(pause)
            plt.close()
