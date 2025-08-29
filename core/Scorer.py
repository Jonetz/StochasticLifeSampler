import numpy as np
import torch
from scipy.stats import entropy
from itertools import combinations

from core.GOLEngine import GoLEngine 

# ---------- SCORERS ----------
class Scorer:
    """Base class for evaluating interestingness."""
    def score(self, trajectory: list) -> float:
        raise NotImplementedError

# ---------- 1. AliveCellCountScorer ----------
class AliveCellCountScorer(Scorer):
    """Score depends only on the current board: fraction of alive cells."""
    def score(self, board: torch.Tensor) -> float:
        b = board if isinstance(board, torch.Tensor) else board.tensor
        return float(b.sum().float() / b.numel())

# ---------- 2. StabilityScorer ----------
class StabilityScorer(Scorer):
    """Rewards temporal stability: fraction of unchanged cells per step."""
    def __init__(self, engine: GoLEngine, steps: int = 10):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def score(self, boards: list[torch.Tensor]) -> torch.Tensor:
        """
        boards: list of initial board tensors (N, H, W)
        Returns: float or tensor of scores per board
        """
        # Batch all boards
        batch = torch.stack(boards, dim=0).to(self.engine.device)
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = torch.stack(traj, dim=0)  # (T, N, H, W)
        diffs = (traj_t[1:] == traj_t[:-1]).float()  # (T-1, N, H, W)
        sims = diffs.mean(dim=(0,2,3))  # mean over time + spatial dims → per board
        return sims.cpu().numpy() if sims.numel() > 1 else float(sims)

# ---------- 3. ChangeRateScorer ----------
class ChangeRateScorer(Scorer):
    """Rewards temporal variability: fraction of changed cells per step."""
    def __init__(self, engine: GoLEngine, steps: int = 10):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def score(self, boards: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.stack(boards, dim=0).to(self.engine.device)
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = torch.stack(traj, dim=0)  # (T, N, H, W)
        flips = (traj_t[1:] != traj_t[:-1]).float()
        rates = flips.mean(dim=(0,2,3))
        return rates.cpu().numpy() if rates.numel() > 1 else float(rates)

# ---------- 4. EntropyScorer ----------
class EntropyScorer(Scorer):
    def score(self, board: torch.Tensor) -> float:
        b = board if isinstance(board, torch.Tensor) else board.tensor
        alive_frac = b.float().mean()
        alive_frac = torch.clamp(alive_frac, 1e-6, 1-1e-6)
        ent = -(alive_frac*torch.log(alive_frac) + (1-alive_frac)*torch.log(1-alive_frac))
        return float(ent)


# ---------- 5. DiversityScorer ----------
class DiversityScorer(Scorer):
    """Diversity as hash uniqueness over multiple boards is not applicable in single-board scoring.
       We can instead define a proxy: fraction of alive cells as a “diversity metric”."""
    def score(self, board: torch.Tensor) -> float:
        # simple proxy: alive fraction
        b = board if isinstance(board, torch.Tensor) else board.tensor
        alive_frac = b.float().mean()
        return float(alive_frac * (1 - alive_frac))  # max when ~50% alive


# ---------- 6. CompositeScorer ----------
class CompositeScorer(Scorer):
    """Combine AliveCellCount + Entropy + Diversity as single-board proxy."""
    def __init__(self):
        self.scorers = [AliveCellCountScorer(), EntropyScorer(), DiversityScorer()]
    def score(self, board: torch.Tensor) -> float:
        return float(np.mean([s.score(board) for s in self.scorers]))


# ---------- 7. OscillationScorer ----------
class OscillationScorer(Scorer):
    """Detect first repeating state as oscillation period."""
    def __init__(self, engine: GoLEngine, steps: int = 100):
        super().__init__()
        self.engine = engine
        self.steps = steps

    def score(self, boards: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.stack(boards, dim=0).to(self.engine.device)
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = [b.clone() for b in traj]  # list of (N,H,W)
        N = batch.shape[0]
        periods = torch.zeros(N, device=self.engine.device)
        # Compute hashes per board per timestep
        flat_size = batch.shape[1]*batch.shape[2]
        powers = torch.arange(1, flat_size+1, device=self.engine.device, dtype=torch.int64)
        for n in range(N):
            seen = {}
            for t in range(len(traj_t)):
                b = traj_t[t][n].view(-1).to(torch.int64)
                h = (b * powers).sum().item()
                if h in seen:
                    periods[n] = t - seen[h]
                    break
                seen[h] = t
        return periods.cpu().numpy() if periods.numel() > 1 else float(periods)