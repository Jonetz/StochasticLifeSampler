from typing import Union
import torch
from core.Board import Board
from core.GOLEngine import GoLEngine
from core.Scorer import OscillationScorer, Scorer
from utils.encodings import board_hash
import torch
from typing import Union

class MethuselahScorer(Scorer):
    """
    Methuselah scorer:
    - Ignores starting alive cells
    - Rewards patterns that survive long and stabilize with many alive cells
    """
    def __init__(self, engine: 'GoLEngine', steps: int = 500, max_seen: int = 128):
        super().__init__()
        self.engine = engine
        self.steps = steps
        self.max_seen = max_seen

    @staticmethod
    def board_hash(batch: torch.Tensor) -> torch.Tensor:
        """Vectorized hash for a batch of boards (B,H,W) -> (B,)"""
        B, H, W = batch.shape
        flat = batch.view(B, -1).to(torch.int64)
        powers = torch.arange(1, H*W + 1, device=batch.device, dtype=torch.int64)
        return (flat * powers).sum(dim=1)
    
    @torch.no_grad()
    def score(self, batch: Union['Board', torch.Tensor]) -> torch.Tensor:
        """
        batch: (B,H,W) tensor
        returns: (B,) float scores
        Score = position of first stable step (1-based), plus alive cells at final step
        """
        if not torch.is_tensor(batch):
            batch = batch.tensor
        B, H, W = batch.shape
        s = batch.bool()

        # ---------------- SIMULATE TRAJECTORY ----------------
        _, traj_list = self.engine.simulate(s, steps=self.steps, return_trajectory=True)
        traj_t = torch.stack(traj_list, dim=0).bool()  # (steps, B, H, W)
        steps_total = traj_t.shape[0]

        # ---------------- COMPUTE STABILITY ----------------
        diffs = traj_t[1:] != traj_t[:-1]              # (steps-1, B, H, W)
        alive_change = diffs.view(steps_total-1, B, -1).any(dim=2)  # (steps-1, B)
        stable_mask = ~alive_change                     # True where step t -> t+1 is stable

        # Add a fake True row at the end to handle boards that never stabilize
        stable_mask_padded = torch.cat([stable_mask, torch.ones(1, B, device=batch.device, dtype=torch.bool)], dim=0)  # (steps, B)

        # First stable index (1-based)
        first_stable_idx = stable_mask_padded.float().argmax(dim=0) + 1   # (B,)
        
        ever_stable = stable_mask.any(dim=0)           # (B,) True if board stabilizes at some point

        first_stable_idx = first_stable_idx * ever_stable.float()

        # ---------------- FINAL ALIVE ----------------
        finalAlive = traj_t[-1].view(B, -1).sum(dim=1).float()

        # ---------------- SCORE ----------------
        scores = finalAlive + first_stable_idx.float()

        # Exclude oscillators
        osc_periods = OscillationScorer(self.engine, steps=self.steps).score(batch)
        scores[osc_periods > 1] = 0.0
        
        print(scores)
        print(osc_periods)

        """
        print('Stable index: ')
        print(first_stable_idx)
        print('Alive Cells: ')
        print(finalAlive)            
        print('Oscilator Score: ')
        print(osc_periods)
        print('Final Score: ')
        print(scores)
        """

        return scores