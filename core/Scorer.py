from typing import Union
import numpy as np
import torch
from scipy.stats import entropy
from itertools import combinations
import hashlib

from core.Board import Board
from core.GOLEngine import GoLEngine 

DEBUG = False
# ---------- SCORERS ----------
class Scorer:
    """Base class for evaluating interestingness."""
    def score(self, batch: Union[Board, torch.Tensor]) -> float:
        raise NotImplementedError

# ---------- 1. AliveCellCountScorer ----------
class AliveCellCountScorer(Scorer):
    """Batchified: returns fraction of alive cells per board in the batch."""
    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        """
        batch: (N, H, W) tensor, dtype=uint8 or bool
        returns: (N,) tensor of alive fractions
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        batch = batch.float()
        alive_frac = batch.sum(dim=(1, 2)) / (batch.shape[1] * batch.shape[2])
        return alive_frac.cpu()

# ---------- 2. StabilityScorer ----------
class StabilityScorer(Scorer):
    """Rewards temporal stability: fraction of unchanged cells per step."""
    def __init__(self, engine: GoLEngine, steps: int = 10):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        """
        batch: list of initial board tensors (N, H, W)
        Returns: float or tensor of scores per board
        """
        # Batch all batch
        if not torch.is_tensor(batch):
            batch = batch.tensor
        if not batch.device == self.engine.device:
            batch = batch.to(self.engine.device)
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = torch.stack(traj, dim=0)  # (T, N, H, W)
        diffs = (traj_t[1:] == traj_t[:-1]).float()  # (T-1, N, H, W)
        sims = diffs.mean(dim=(0,2,3))  # mean over time + spatial dims → per board
        return sims.cpu()

# ---------- 3. ChangeRateScorer ----------
class ChangeRateScorer(Scorer):
    """Rewards temporal variability: fraction of changed cells per step."""
    def __init__(self, engine: GoLEngine, steps: int = 10):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch.tensor
        if not batch.device == self.engine.device:
            batch = batch.to(self.engine.device)
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = torch.stack(traj, dim=0)  # (T, N, H, W)
        flips = (traj_t[1:] != traj_t[:-1]).float()
        rates = flips.mean(dim=(0,2,3))
        return rates.cpu()

# ---------- 4. EntropyScorer ----------
class EntropyScorer(Scorer):
    """Batchified: entropy per board, dim 0 is batch dimension."""
    def __init__(self, engine: GoLEngine, steps: int = 10):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        """
        batch: (N, H, W) tensor, dtype=uint8 or bool
        returns: (N,) tensor of entropies
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        if not batch.device == self.engine.device:
            batch = batch.to(self.engine.device)
        batch = batch.float()
        alive_frac = batch.mean(dim=(1, 2))  # fraction of alive cells per board
        alive_frac = torch.clamp(alive_frac, 1e-6, 1-1e-6)
        ent = -(alive_frac * torch.log(alive_frac) + (1 - alive_frac) * torch.log(1 - alive_frac))
        return ent.cpu()


# ---------- 5. DiversityScorer ----------
class DiversityScorer(Scorer):
    """Batchified diversity proxy: alive fraction * (1-alive fraction) per board."""
    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        """
        batch: (N, H, W) tensor
        returns: (N,) tensor of diversity scores
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        batch = batch.float()
        alive_frac = batch.mean(dim=(1, 2))
        return (alive_frac * (1 - alive_frac)).cpu()



# ---------- 6. CompositeScorer ----------
# TODO Make this with scorers as parameters
#class CompositeScorer(Scorer):
#    """Combine AliveCellCount + Entropy + Diversity as single-board proxy."""
#    def __init__(self):
#        self.scorers = [AliveCellCountScorer(), DiversityScorer()]
#    def score(self, board: torch.Tensor) -> float:
#        return float(np.mean([s.score(board) for s in self.scorers]))


# ---------- 7. OscillationScorer ----------
class OscillationScorer(Scorer):
    """Detect first repeating state as oscillation period."""
    def __init__(self, engine: GoLEngine, steps: int = 100):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def _hash_board(self, board: torch.Tensor) -> str:
        """
        Stable hash of a single board state.
        Proceed with caution

        board: (H,W) binary tensor
        returns: hex digest string
        """
        return hashlib.sha1(board.cpu().numpy().tobytes()).hexdigest()

    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch.tensor
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = [b.clone() for b in traj]  # list of (N,H,W)
        N = batch.shape[0]
        periods = torch.zeros(N, device=self.engine.device)

        for n in range(N):
            seen = {}
            for t in range(len(traj_t)):
                h = self._hash_board(traj_t[t][n])
                if h in seen:
                    periods[n] = t - seen[h]
                    if DEBUG:
                        print(f"[Board {n}] Repeat found at step {t}, "
                              f"period = {periods[n].item()}, "
                              f"first seen at step {seen[h]}")
                    break
                seen[h] = t

            if DEBUG and periods[n] == 0:
                print(f"[Board {n}] No repeat found within {self.steps} steps")

        return periods.cpu()