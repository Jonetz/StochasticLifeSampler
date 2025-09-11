from typing import Union
import torch
from core.Board import Board
from core.Scorer import OscillationScorer, Scorer
import torch
from typing import Union
import torch.nn.functional as F

class MethuselahScorer(Scorer):
    """
    Methuselah scorer:
    - Ignores starting alive cells
    - Rewards patterns that survive long and stabilize with many alive cells
    """
    @staticmethod
    def board_hash(batch: torch.Tensor) -> torch.Tensor:
        """Vectorized hash for a batch of boards (B,H,W) -> (B,)"""
        B, H, W = batch.shape
        flat = batch.view(B, -1).to(torch.int64)
        powers = torch.arange(1, H*W + 1, device=batch.device, dtype=torch.int64)
        return (flat * powers).sum(dim=1)
    
    @torch.no_grad()
    def score(self, batch: Union['Board', torch.Tensor], final_board: Union['Board', torch.Tensor]= None) -> torch.Tensor:
        """
        batch: (B,H,W) tensor
        returns: (B,) float scores
        Score = position of first stable step (1-based), plus alive cells at final step
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        B, H, W = batch.shape
        s = batch.bool()

        # ---- simulate with stability detection ----
        final_board, first_stable_idx = self.engine.simulate(
            s, steps=self.steps, return_stability=True
        )
        # Final alive count
        finalAlive = final_board.tensor.view(B, -1).sum(dim=1).float()

        # Scores = alive*3 + first_stable_idx
        scores = finalAlive * 2 + first_stable_idx.float()

        # Exclude oscillators
        osc_periods = OscillationScorer(self.engine, steps=self.steps).score(batch)
        scores[osc_periods > 5] = 0.0

        return scores * 5e-2

class RotInvariantScorer:
    """
    Scorer that rewards patterns that are rotationally invariant
    across their trajectory and end with many alive cells.
    """

    def __init__(self, engine, steps: int = 512, kernel: torch.Tensor = None):
        self.engine = engine
        self.steps = steps

        # Precompute kernel for hashing if not provided
        if kernel is None:
            self.kernel = torch.randint(
                1, 2**16,
                size=(1, 1, 3, 3),  # 3x3 kernel, can make bigger
                device=engine.device,
                dtype=torch.float32,  # must be float for conv2d
            )
        else:
            self.kernel = kernel.to(dtype=torch.float32, device=engine.device)

    @torch.no_grad()
    def rotational_invariant_hash(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute rotationally invariant hashes for a batch of boards.

        Args:
            states: (B,H,W) bool/uint8 tensor

        Returns:
            hashes: (B,) int64 tensor
        """
        B, H, W = states.shape
        s = states.unsqueeze(1).float()  # (B,1,H,W)

        def conv_hash(x):
            conv_out = F.conv2d(x, self.kernel, padding=self.kernel.shape[-1]//2)
            return conv_out.view(B, -1).sum(dim=1).to(torch.int64)

        # All 4 rotations
        h0   = conv_hash(s)
        h90  = conv_hash(s.rot90(1, dims=(2, 3)))
        h180 = conv_hash(s.rot90(2, dims=(2, 3)))
        h270 = conv_hash(s.rot90(3, dims=(2, 3)))

        return torch.min(torch.min(h0, h90), torch.min(h180, h270))

    @torch.no_grad()
    def score(self, batch: Union["Board", torch.Tensor], debug: bool = False) -> torch.Tensor:
        """
        Score batch of boards.

        Args:
            batch: (B,H,W) tensor
            debug: if True, prints debug info for first board

        Returns:
            scores: (B,) tensor
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        B, H, W = batch.shape

        # Run trajectory
        _, traj = self.engine.simulate(batch, steps=self.steps, return_trajectory=True)
        traj_t = torch.stack(traj, dim=0).bool()  # (steps, B, H, W)

        # Hash trajectory with rotational invariance
        hashes = []
        for t in range(traj_t.shape[0]):
            hashes.append(self.rotational_invariant_hash(traj_t[t]))
        hashes = torch.stack(hashes, dim=0)  # (steps, B)

        # Detect oscillations
        periods = torch.zeros(B, device=batch.device)
        for b in range(B):
            seen = {}
            for t in range(self.steps):
                h = hashes[t, b].item()
                if h in seen:
                    periods[b] = t - seen[h]
                    break
                seen[h] = t

        # Final alive counts
        finalAlive = traj_t[-1].view(B, -1).sum(dim=1).float()

        # Score = alive cells if no oscillation, else 0
        scores = finalAlive.clone()
        scores[periods > 0] = 0.0

        if debug:
            b = 0
            print(f"[DEBUG] Board {b}:")
            print(f"  Final alive: {finalAlive[b].item()}")
            print(f"  Period: {periods[b].item()}")
            print(f"  Score: {scores[b].item()}")

        return scores

# ---------- CombinedScorer ----------
class CombinedScorer(Scorer):
    """
    Combines multiple child scorers with specified weights.

    Args:
        scorers: List of tuples (Scorer, weight).
        normalize: If True, normalize weights to sum=1.
        reduction: 'sum' or 'mean' for combining scores.

    Methods:
        score(batch): Returns weighted combination of child scores.

    Safeguards:
        - All scorers must return tensors of shape (N,)
        - Normalizes weights if requested
        - Works batch-wise on (N,H,W) tensors
    """
    def __init__(self, engine, scorers: list[tuple[Scorer, float]], normalize: bool = True, reduction: str = "sum", steps: int = 1):
        super().__init__(engine)
        self.scorers = [s for s, _ in scorers]
        weights = torch.tensor([w for _, w in scorers], dtype=torch.float32, device=engine.device)
        if normalize:
            weights = weights / weights.sum()
        self.weights = weights
        assert reduction in ("sum", "mean"), "reduction must be 'sum' or 'mean'"
        self.reduction = reduction
        self.steps = steps

    def score(self, batch: Union[Board, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch.tensor
        if not batch.device == self.engine.device:
            batch = batch.to(self.engine.device)

        final_board = self.engine.simulate(batch, steps=self.steps) 
        scores = []
        for scorer in self.scorers:
            scores.append(scorer.score(batch, final_board=final_board))  # (N,)
        scores = torch.stack(scores, dim=1)  # (N, num_scorers)

        weighted = scores * self.weights.view(1, -1)

        if self.reduction == "sum":
            return weighted.sum(dim=1)  # (N,)
        else:  # mean
            return weighted.mean(dim=1)  # (N,)
