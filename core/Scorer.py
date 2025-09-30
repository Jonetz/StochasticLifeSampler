from typing import Optional, Union
import torch
import hashlib

from core.Board import Board
from core.GOLEngine import GoLEngine
from utils.encodings import board_hash_weighted 

DEBUG = False
# ---------- SCORERS ----------
class Scorer:
    """
    Base class for evaluating board interestingness in a batch.

    Args:
        engine: GoLEngine instance used for simulation.
        steps: Number of simulation steps for scoring (optional).

    Attributes:
        name: Name of the scorer class for identification.
    """
    def __init__(self, engine: GoLEngine, steps: Optional[int] = 100):
        self.engine = engine
        self.steps = steps
        self.name = self.__class__.__name__ 

    def score(self,batch: Union[Board, torch.Tensor]):
        """
        Compute score for a batch of boards.

        Args:
            batch: Board or tensor of shape (N,H,W).

        Returns:
            Tensor of scores, shape (N,).

        Safeguards:
            Must be implemented in subclass.
        """
        raise NotImplementedError

# ---------- 1. AliveCellCountScorer ----------
class AliveCellCountScorer(Scorer):
    """
    Returns fraction of alive cells per board after simulation.

    Args:
        batch: Board or tensor (N,H,W).

    Returns:
        Tensor of shape (N,) with fractions in [0,1].

    Safeguards:
        - Converts input to float.
        - Runs simulation on engine before counting.
    """
    def score(self, batch: Union[Board, torch.Tensor], final_board: Union[Board, torch.Tensor] = None ) -> torch.Tensor:
        """
        batch: (N, H, W) tensor, dtype=uint8 or bool
        returns: (N,) tensor of alive fractions
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        batch = batch.float()        
        batch = self.engine.simulate(batch, steps=self.steps) if final_board is None else final_board
        alive_count = batch._states.sum(dim=(1, 2)) #/ (batch._states.shape[1] * batch._states.shape[2])
        return alive_count.float() * 0.3
    
# ---------- 2. StabilityScorer ----------
class StabilityScorer(Scorer):
    """
    Rewards temporal stability: fraction of unchanged cells across steps.

    Args:
        batch: Board or tensor (N,H,W).

    Returns:
        Tensor of shape (N,) giving mean fraction of unchanged cells per board.

    Safeguards:
        - Uses engine.simulate with return_trajectory=True.
        - Converts input to proper device and dtype.
    """
    def score(self, batch: Union[Board, torch.Tensor], final_board: Union[Board, torch.Tensor] = None ) -> torch.Tensor:
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
        return sims

# ---------- 3. ChangeRateScorer ----------
class ChangeRateScorer(Scorer):
    """
    Rewards temporal variability: fraction of cells that changed state.

    Args:
        batch: Board or tensor (N,H,W).

    Returns:
        Tensor of shape (N,) with fraction of changed cells per board.

    Safeguards:
        - Ensures tensors are on correct device.
        - Computes element-wise difference correctly for batch.
    """
    def score(self, batch: Union[Board, torch.Tensor], final_board: Union[Board, torch.Tensor] = None ) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch.tensor
        if not batch.device == self.engine.device:
            batch = batch.to(self.engine.device)        
        batch = self.engine.simulate(batch, steps=self.steps) if final_board is None else final_board
        batch_x = self.engine.simulate(batch, steps=1)    
        # Compute elementwise difference
        diff = batch._states != batch_x._states
        # Fraction of changed cells per board
        rates = diff.to(torch.float32).view(diff.shape[0], -1).sum(dim=1)

        return rates * 0.5

# ---------- 4a. EntropyScorer ----------
class EntropyScorer(Scorer):
    """
    Computes binary entropy of alive fraction per board.

    Args:
        batch: Board or tensor (N,H,W).

    Returns:
        Tensor of shape (N,) with entropy values.

    Safeguards:
        - Clamps alive fraction to avoid log(0).
        - Works batch-wise on (N,H,W) tensors.
    """
    def score(self, batch: Union[Board, torch.Tensor], final_board: Union[Board, torch.Tensor] = None ) -> torch.Tensor:
        """
        batch: (N, H, W) tensor, dtype=uint8 or bool
        returns: (N,) tensor of entropies
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        if not batch.device == self.engine.device:
            batch = batch.to(self.engine.device)
        batch = self.engine.simulate(batch, steps=self.steps) if final_board is None else final_board
        batch = batch.float()
        alive_frac = batch.mean(dim=(1, 2))  # fraction of alive cells per board
        alive_frac = torch.clamp(alive_frac, 1e-6, 1-1e-6)
        ent = -(alive_frac * torch.log(alive_frac) + (1 - alive_frac) * torch.log(1 - alive_frac))
        return ent

# ---------- 4b. ChaosScorer ----------
class ChaosScorer(Scorer):
    """
    Estimates "chaoticity" of Game of Life boards.

    Computes metrics over a trajectory:
      - Entropy of alive fraction
      - Variance of bounding box size
      - Temporal uniqueness via state hashes

    Args:
        steps: Number of warmup generations before evaluation.
        weight_entropy: Weight for alive-fraction entropy.
        weight_growth: Weight for bounding-box growth variance.
        weight_uniqueness: Weight for temporal uniqueness.
        eval_steps: Number of evaluation steps after warmup/final board.
    """
    def __init__(self, engine, 
                 steps: int = 5,
                 weight_entropy: float = 1.0,
                 weight_growth: float = 1.0,
                 weight_uniqueness: float = 1.0,
                 eval_steps: int = 50):
        super().__init__(engine)
        self.steps = steps
        self.w_entropy = weight_entropy
        self.w_growth = weight_growth
        self.w_unique = weight_uniqueness
        self.eval_steps = eval_steps

    def score(self, 
              batch: Union[Board, torch.Tensor], 
              final_board: Union[Board, torch.Tensor] = None) -> torch.Tensor:
        """
        Compute chaoticity score for a batch of boards.

        Args:
            batch: Initial board batch.
            final_board: If provided, skip warmup and start from here.

        Returns:
            Tensor of shape (N,) with scores for each board in the batch.
        """
        # --- Normalize input ---
        if not torch.is_tensor(batch):
            batch = batch.tensor
        batch = batch.to(self.engine.device, dtype=torch.float)
        N, H, W = batch.shape

        # --- Warmup or use final board ---
        if final_board is not None:
            if isinstance(final_board, Board):
                start = final_board.tensor.float().to(self.engine.device)
            else:
                start = final_board.to(self.engine.device, dtype=torch.float)
        else:
            start = self.engine.simulate(batch, steps=self.steps).tensor.float()

        # --- Simulate trajectory ---
        _, traj = self.engine.simulate(start, steps=self.eval_steps, return_trajectory=True)
        traj = torch.stack(traj).to(self.engine.device).to(torch.float32)  # (eval_steps, N, H, W)

        # --- Entropy of alive fraction ---
        alive_frac = traj.mean(dim=(2, 3))  # (eval_steps, N)
        alive_frac = torch.clamp(alive_frac, 1e-6, 1 - 1e-6)
        entropies = -(alive_frac * torch.log(alive_frac) +
                      (1 - alive_frac) * torch.log(1 - alive_frac))
        ent_score = entropies.mean(dim=0)  # (N,)

        # --- Bounding box variance ---
        mask = traj > 0.5
        ys = mask.any(dim=3).float().sum(dim=2)  # height
        xs = mask.any(dim=2).float().sum(dim=2)  # width
        growth_var = (ys * xs).var(dim=0)

        # --- Temporal uniqueness (hash-based) ---
        coords = torch.arange(H * W, device=traj.device, dtype=torch.float).view(1, 1, H, W)
        hashes = (traj * coords).sum(dim=(2, 3))  # (eval_steps, N)
        uniqueness_score = hashes.std(dim=0)

        # --- Weighted sum ---
        total_score = (
            self.w_entropy * ent_score +
            self.w_growth * growth_var +
            self.w_unique * uniqueness_score
        )
        return total_score * 5e-5

# ---------- 5. DiversityScorer ----------
class DiversityScorer(Scorer):
    """
    Computes diversity proxy: alive fraction * (1-alive fraction).

    Args:
        batch: Board or tensor (N,H,W).

    Returns:
        Tensor of shape (N,) with diversity scores.

    Safeguards:
        - Converts tensor to float.
        - Returns CPU tensor.
    """
    def score(self, batch: Union[Board, torch.Tensor], final_board: Union[Board, torch.Tensor] = None ) -> torch.Tensor:
        """
        batch: (N, H, W) tensor
        returns: (N,) tensor of diversity scores
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        batch = self.engine.simulate(batch, steps=self.steps).tensor.float() if final_board is None else final_board
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
    """
    Detects first repeating board configuration to estimate oscillation period.

    Methods:
        _hash_board: Stable hash of a single board.
        score: Returns oscillation period per board.

    Args:
        batch: Board or tensor (N,H,W).

    Returns:
        Tensor of shape (N,) with oscillation periods (0 if no repeat).

    Safeguards:
        - Uses hashlib.sha1 for stable hashing.
        - Temporarily sets engine step_fn for hashing.
        - Restores step_fn after scoring.
        - DEBUG flag prints warning if no repeat found.
    """
    def __init__(self, engine, steps = 100):
        super().__init__(engine, steps)
        self.kernel = torch.randint(1, 2**16, size=(3, 3), device=self.engine.device, dtype=torch.int64).float()

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
        N = batch.shape[0]
        periods = torch.zeros(N, device=self.engine.device)        
        self.engine.set_step_fn(
            lambda s, t: board_hash_weighted(s, self.kernel)
        )
        _, traj = self.engine.simulate(batch, steps=self.steps)

        self.engine.set_step_fn(None)

        for n in range(N):
            seen = {}
            for t, hashes in enumerate(traj):
                h = hashes[n]
                if h in seen:
                    periods[n] = t - seen[h]
                    break
                seen[h] = t

            if DEBUG and periods[n] == 0:
                print(f"[Board {n}] No repeat found within {self.steps} steps")

        return periods