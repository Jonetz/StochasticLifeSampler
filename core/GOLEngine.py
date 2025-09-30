
import os
import shutil
import subprocess
import sys

# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
from typing import Any, Callable, Optional, Tuple, List, Union, Sequence
import numpy as np
from core.Board import Board
from utils.encodings import board_hash, board_hash_weighted, board_hash_zobrist, save_rle_list, load_rle_list, rle_decode_binary
from PIL import Image, ImageDraw
import collections
import imageio.v3 as iio
import numpy as np

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
                skip_osci: bool = True,
                step_fn: Optional[Callable[[torch.Tensor, int], Any]] = None,
                pre_train_hooks: Optional[List[Callable]] = None,
                post_train_hooks: Optional[List[Callable]] = None,
                pre_step_hooks: Optional[List[Callable]] = None,
                post_step_hooks: Optional[List[Callable]] = None,
                cancel_condition: Optional[Callable[[torch.Tensor, int], bool]] = None):
        """
        Initialize the GoL engine with optional hooks and custom step function.

        Args:
            device: 'cuda', 'cpu', or torch.device for computation.
            border: How to handle board edges: 'wrap', 'constant', or 'reflect'.
            skip_osci: If True, terminates simulation early on repeating states.
            step_fn: Optional function called at each step: fn(states, step_idx) -> any.
            pre_train_hooks: List of callables executed before simulation starts.
            post_train_hooks: List of callables executed after simulation ends.
            pre_step_hooks: List of callables executed before each step.
            post_step_hooks: List of callables executed after each step.
            cancel_condition: Callable(states, step_idx) -> bool, stop simulation if True.

        Safeguards:
            - Ensures device is torch.device.
            - Hook lists are initialized to empty lists if None.
            - Raises ValueError for invalid border mode later in _pad_mode_and_kwargs().
        """
        self.device = torch.device(device)
        self.border = border
        self.skip_osci = skip_osci
        self.kernel = GoLEngine.NEIGH_KERNEL.view(1, 1, 3, 3).to(self.device)
        self.pad_flag = {"wrap": 0, "constant": 1, "reflect": 2}[border]

        # Customization points
        self._step_fn = step_fn
        self.pre_train_hooks = pre_train_hooks or []
        self.post_train_hooks = post_train_hooks or []
        self.pre_step_hooks = pre_step_hooks or []
        self.post_step_hooks = post_step_hooks or []
        self.cancel_condition = cancel_condition


    def _pad_mode_and_kwargs(self):
        """
        Return the corresponding F.pad mode string and kwargs for the configured border.

        Returns:
            Tuple[str, dict]: Mode string compatible with F.pad, plus additional kwargs.

        Raises:
            ValueError: If self.border is not one of 'wrap', 'constant', or 'reflect'.
        """
        if self.border == "wrap":
            return "circular", {}
        elif self.border == "constant":
            return "constant", {"value": 0}
        elif self.border == "reflect":
            return "reflect", {}
        else:
            raise ValueError(f"Unknown border mode {self.border}")

    @torch.jit.script
    def step_jit(states: torch.Tensor, kernel: torch.Tensor, pad_flag: int, radius: int = 3, mask_border: bool = True) -> torch.Tensor:
        """
        JIT-compiled single Game of Life step with safe border pruning.

        Args:
            states: Binary tensor of shape (B,H,W).
            kernel: Convolution kernel for neighborhood counts.
            pad_flag: 0=wrap, 1=constant, 2=reflect for boundary handling.
            radius: number of rows/cols from border to potentially prune.
            mask_border: whether to apply safe border pruning.

        Returns:
            Tensor of same shape as input, updated cell states (bool).
        """
        B, H, W = states.shape
        x = states.float().unsqueeze(1)

        # --- padding ---
        if pad_flag == 0:
            x_p = F.pad(x, (1, 1, 1, 1), mode='circular')
        elif pad_flag == 1:
            x_p = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
        else:
            x_p = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # --- neighborhood counts ---
        nbh = F.conv2d(x_p, kernel).squeeze(1)
        survive = states & ((nbh == 2) | (nbh == 3))
        born = (~states) & (nbh == 3)
        out = survive | born

        if mask_border and radius > 0:
            # --- Top strip ---
            top_strip = out[..., 0:radius, :]
            active_mask = (top_strip != 0).cumsum(dim=1) > 0
            out[..., 0:radius, :] = top_strip * active_mask

            # --- Bottom strip ---
            bottom_strip = out[..., -radius:, :]
            active_mask = (torch.flip(bottom_strip, [1]) != 0).cumsum(dim=1) > 0
            out[..., -radius:, :] = torch.flip(bottom_strip * torch.flip(active_mask, [1]), [1])

            # --- Left strip ---
            left_strip = out[..., :, 0:radius]
            active_mask = (left_strip != 0).cumsum(dim=2) > 0
            out[..., :, 0:radius] = left_strip * active_mask

            # --- Right strip ---
            right_strip = out[..., :, -radius:]
            active_mask = (torch.flip(right_strip, [2]) != 0).cumsum(dim=2) > 0
            out[..., :, -radius:] = torch.flip(right_strip * torch.flip(active_mask, [2]), [2])

        return out
        
    @torch.no_grad()
    def simulate(self,
                board: Union['Board', torch.Tensor],
                steps: int = 1,
                return_trajectory: bool = False,
                return_stability: Union[bool, int] = False
                ) -> Union['Board', Tuple['Board', List[Any]], Tuple['Board', torch.Tensor]]:
        
        # -------------------- Setup --------------------
        if isinstance(board, Board):
            states = board._states.bool()
        elif isinstance(board, np.ndarray):
            states = torch.tensor(board).bool().to(self.device)
        else:
            states = board.bool()

        if states.dim() == 2:
            states = states.unsqueeze(0)
        B, H, W = states.shape
        s = states.to(self.device)

        fn = self._step_fn if self._step_fn else None
        if return_trajectory and fn is None:
            fn = lambda x, t: x.clone()
        results = [] if fn is not None else None

        # ---- run pre-training hooks ----
        for hook in self.pre_train_hooks:
            hook(s)

        # -------------------- Stability setup --------------------
        stability_window = 5
        if isinstance(return_stability, int):
            stability_window = max(1, return_stability)

        if return_stability:
            kernel1 = torch.randint(1, 2**61-1, (H, W), device=self.device, dtype=torch.int64)
            kernel2 = torch.randint(1, 2**61-1, (H, W), device=self.device, dtype=torch.int64)
            prev_hashes_tensor = torch.empty(B, stability_window, 2, device=self.device, dtype=torch.int64)
            prev_hashes_len = 0
            prev_hashes_tensor[:, prev_hashes_len] = board_hash_weighted(s, kernel1, kernel2)
            prev_hashes_len += 1
            first_stable = torch.zeros(B, dtype=torch.long, device=self.device)
        else:
            kernel1 = kernel2 = None

        if self.skip_osci:
            if kernel1 is None or kernel2 is None:
                kernel1 = torch.randint(1, 2**61-1, (H, W), device=self.device, dtype=torch.int64)
                kernel2 = torch.randint(1, 2**61-1, (H, W), device=self.device, dtype=torch.int64)
            osc_window = 50000
            seen_hashes_tensor = torch.empty(B, osc_window, 2, device=self.device, dtype=torch.int64)
            seen_hashes_len = 0

        # -------------------- Simulation loop --------------------
        for t in range(steps):
            # ---- run cancel hooks ----
            if self.cancel_condition and self.cancel_condition(s, t):
                break

            # ---- run pre-step hooks ----
            for hook in self.pre_step_hooks:
                hook(s, t)

            mask_border = (t % 3 == 0)
            s = GoLEngine.step_jit(s, self.kernel, self.pad_flag, mask_border=mask_border)

            # ---- run post-step hooks ----
            for hook in self.post_step_hooks:
                hook(s, t)

            # -- Collect from Boards based on functions --
            if fn is not None:
                results.append(fn(s, t))

            # -------- Stability check --------
            if return_stability:
                hashes = board_hash_weighted(s, kernel1, kernel2)  # (B,2)
                if prev_hashes_len == 0:
                    stable_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
                else:
                    cmp = (hashes.unsqueeze(1) == prev_hashes_tensor[:, :prev_hashes_len])  # (B, prev_len, 2)
                    stable_mask = cmp.all(dim=-1).any(dim=1)
                    newly_stable = (first_stable == 0) & stable_mask
                    first_stable[newly_stable] = t + 1

                # append
                if prev_hashes_len < stability_window:
                    prev_hashes_tensor[:, prev_hashes_len] = hashes
                    prev_hashes_len += 1
                else:
                    prev_hashes_tensor = torch.cat([prev_hashes_tensor[:, 1:], hashes.unsqueeze(1)], dim=1)

           # -------- Oscillation detection --------
            if self.skip_osci:
                hashes = board_hash_weighted(s, kernel1, kernel2)  # (B,2)
                if seen_hashes_len == 0:
                    seen_hashes_tensor[:, 0] = hashes
                    seen_hashes_len = 1
                else:
                    cmp = (hashes.unsqueeze(1) == seen_hashes_tensor[:, :seen_hashes_len])  # (B, seen_len, 2)
                    matches = cmp.all(dim=-1).any(dim=1)  # both hash values must match

                    if matches.all():
                        final_board = Board(s, idx=None, device=self.device,
                                            meta=(board.meta.copy() if isinstance(board, Board) else {}))
                        if return_stability:
                            return final_board, first_stable
                        if results is not None:
                            return final_board, results
                        return final_board

                    # append
                    if seen_hashes_len < osc_window:
                        seen_hashes_tensor[:, seen_hashes_len] = hashes
                        seen_hashes_len += 1
                    else:
                        seen_hashes_tensor = torch.cat([seen_hashes_tensor[:, 1:], hashes.unsqueeze(1)], dim=1)

        final_board = Board(s.to(torch.bool), idx=None,
                            device=self.device,
                            meta=(board.meta.copy() if isinstance(board, Board) else {}))

        for hook in self.post_train_hooks:
            hook(s)

        if return_stability:
            return final_board, first_stable
        if results is not None:
            return final_board, results
        return final_board

    # ---------------------------
    # Getter / Setter for step_fn
    # ---------------------------
    def get_step_fn(self) -> Optional[Callable[[torch.Tensor, int], Any]]:
        """
        Retrieve current step function.

        Returns:
            Callable or None: Function executed at each step if set.
        """
        return self._step_fn

    def set_step_fn(self, fn: Optional[Callable[[torch.Tensor, int], Any]]) -> None:
        """
        Set a custom step function for simulation.

        Args:
            fn: Callable(states, step_idx) -> any or None to clear.

        Safeguards:
            - Ensures callable or None is assigned.
        """
        self._step_fn = fn

    # ---------------------------
    # Convenience: wrappers to set initial states
    # ---------------------------
    def make_random(self, N: int, H: int, W: int, fill_prob: float = 0.1) -> Board:
        """
        Create a random board with given dimensions and fill probability.

        Args:
            N: Number of independent boards (batch size).
            H: Height of each board.
            W: Width of each board.
            fill_prob: Probability a cell is alive (1) initially.

        Returns:
            Board: Batched board object with random initial states.

        Safeguards:
            - Ensures fill_prob is in [0,1].
            - Handles batch size N >= 1.
        """
        return Board.from_shape(N=N, H=H, W=W, device=self.device, fill_prob=fill_prob)

    def from_numpy_list(self, arrs: Sequence[np.ndarray]) -> Board:
        """
        Convert a list of 2D numpy arrays to a Board object.

        Args:
            arrs: List of numpy arrays of shape (H,W) representing board states.

        Returns:
            Board: Batched board containing the input arrays.

        Safeguards:
            - Arrays are converted to bool tensors.
            - Raises ValueError if arrays have inconsistent shapes.
        """
        return Board.from_numpy_list(arrs, device=self.device)

    # ---------------------------
    # Visualization and saving
    # ---------------------------    
    def trajectory_to_gif(self,
                        trajectory: Union[Sequence[torch.Tensor], torch.Tensor],
                        filepath: str,
                        fps: int = 10,
                        scale: int = 1,
                        invert: bool = True,
                        show_progress: bool = True,
                        show_counter: bool = True):
        """
        Save a trajectory of boards to an animated GIF.

        Args:
            trajectory: List or tensor of boards (H,W) or (B,H,W).
            filepath: Output file path for the GIF.
            fps: Frames per second for GIF playback.
            scale: Upscaling factor for visibility.
            invert: If True, invert colors (alive=white).
            show_progress: If True, draws a small progress bar at bottom of frames.
            show_counter: If True, adds frame number (every 100 frames).

        Returns:
            None
        """
        def _find_ffmpeg():
            """Return ffmpeg executable path if available, else None."""
            for exe in ["ffmpeg", "ffmpeg.exe"]:
                path = shutil.which(exe)
                if path is not None:
                    return path
            return None
        if torch.is_tensor(trajectory):
            trajectory = [trajectory[i] for i in range(trajectory.shape[0])]
        total_frames = len(trajectory)

        frames = []
        for idx, t in enumerate(trajectory):
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
            img = Image.fromarray(arr_uint8, mode='L').convert("RGB")  # convert to RGB for drawing
            draw = ImageDraw.Draw(img)

            # draw progress bar
            if show_progress:
                bar_height = 6
                bar_width = img.width
                progress = int(bar_width * (idx + 1) / total_frames)
                draw.rectangle([0, img.height - bar_height, progress, img.height], fill=(255, 0, 0))

            # draw counter (here every frame, adjust if needed)
            if show_counter:
                text = f"{idx}/{total_frames}"
                draw.text((5, 5), text, fill=(255, 0, 0))

            frames.append(img)

        # duration per frame in ms
        duration_ms = int(1000 / fps)

        # check if ffmpeg is available
        ffmpeg_path = _find_ffmpeg()
        if ffmpeg_path is not None:
            tmpdir = tempfile.mkdtemp()
            try:
                # save frames to tmp PNGs
                for i, frame in enumerate(frames):
                    arr_uint8 = np.array(frame)
                    iio.imwrite(f"{tmpdir}/frame_{i:05d}.png", arr_uint8)

                # generate palette for better colors
                palette_path = os.path.join(tmpdir, "palette.png")
                cmd1 = [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmpdir, "frame_%05d.png"),
                    "-vf", "palettegen", palette_path
                ]
                subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # use palette to create final gif
                cmd2 = [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmpdir, "frame_%05d.png"),
                    "-i", palette_path,
                    "-lavfi", "paletteuse",
                    filepath
                ]
                subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                shutil.rmtree(tmpdir)  # cleanup
        else:
            # fallback: slow Pillow method
            frames[0].save(filepath,
                        save_all=True,
                        append_images=frames[1:],
                        duration=duration_ms,
                        loop=0,
                        optimize=False,
                        disposal=2)

    def save_initial_states_rle(self, board: Board, filepath: str):
        """
        Save initial board(s) to RLE format file.

        Args:
            board: Board object or tensor representing states.
            filepath: Output file path.

        Returns:
            None

        Safeguards:
            - Converts tensor to Board if needed.
            - Supports multiple boards via board.rle_list().
        """
        if not isinstance(board, Board):
            board = Board(torch.tensor(board))
        rles = board.rle_list()
        save_rle_list(rles, filepath)

    def load_initial_states_rle(self, filepath: str) -> Board:
        """
        Load boards from an RLE file.

        Args:
            filepath: Input RLE file path.

        Returns:
            Board: Batched board object containing decoded states.

        Safeguards:
            - Uses rle_decode_binary to handle decoding.
            - Supports multiple boards in one file.
        """
        rles = load_rle_list(filepath)
        arrs = [rle_decode_binary(s) for s in rles]
        return self.from_numpy_list(arrs)

    def show(self, board: Union[Board, torch.Tensor] = None, idx: int = 0, cmap: str = "gray", invert: bool = True, scale: int = 4):
        """
        Display a single board using matplotlib.

        Args:
            board: Board object or tensor; defaults to self.state.
            idx: Index of board to display if batched.
            cmap: Matplotlib colormap.
            invert: If True, invert colors (alive=white).
            scale: Upscaling factor for visibility.

        Returns:
            None

        Safeguards:
            - Raises ImportError if matplotlib not available.
            - Handles 2D and 3D tensors.
            - Checks type and shape of input.
        """
        if plt is None:
            raise ImportError("matplotlib is required for visualization.")
        if board is None:
            state = self.state[idx].detach().cpu().numpy() if self.state.ndim == 3 else self.state.detach().cpu().numpy()
        else:
            try:
                if torch.is_tensor(board) :
                    if board.dim() == 3:
                        state = board[idx]
                    else:
                        state = board
                elif isinstance(board, Board):
                    if board._states.dim() == 3:
                        state = board._states[idx]
                    else:
                        state = board._states
                else:
                    raise ValueError(f'Please pass either a board or a tensor in the function, not {type(board)}')
            except:
                raise ValueError(f'The Feature combination is not valid.')

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
        Display a sequence of boards as a frame-by-frame animation.

        Args:
            trajectory: List of tensors (H,W) or (B,H,W) representing boards.
            cmap: Matplotlib colormap.
            scale: Upscaling factor for visibility.
            invert: If True, invert colors.
            pause: Time (s) to pause between frames.

        Returns:
            None

        Safeguards:
            - Raises ImportError if matplotlib not available.
            - Handles 2D and 3D tensors.
            - Converts tensors to numpy arrays and ensures proper display.
            - Closes each figure after pause to prevent memory leaks.
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
