
try:
    import torch
except Exception as e:
    raise ImportError("This module requires PyTorch (torch). Install PyTorch and retry.") from e

import math
from typing import Optional
from core.Board import Board
from core.Scorer import Scorer
from core.Proposal import Proposal
from mcmc.Scheduler import BaseTemperatureScheduler

class Chain:
    """
    A single MCMC chain for batched Game of Life boards.

    Each chain manages a batch of boards and performs Metropolis-Hastings
    updates according to a given Proposal and Scorer. Supports optional
    adaptive step size adjustment.

    Args:
        init_board: Board, initial configuration (batch shape N,H,W)
        scorer: Scorer object to evaluate board "fitness"
        proposal: Proposal object to generate candidate boards
        temperature: float, controls MH acceptance probability
        adaptive_steps: bool, whether to adapt step size
        min_steps: minimum number of simulation steps
        max_steps: maximum number of simulation steps
        increase_factor: factor for step increment if adaptation triggers
        patience: number of recent steps to compute improvement over
    """
    def __init__(self, 
                 init_board: Board, 
                 scorer: Scorer,
                 proposal: Proposal,
                 temperature: float = 1.0,
                 scheduler: BaseTemperatureScheduler = None,
                 adaptive_steps: bool = False,
                 min_steps: int = 64,
                 max_steps: int = 4096,
                 increase_factor: float = 1.2,
                 patience: int = 5):
    
        self.board = init_board        # Board with shape (N,H,W)
        self.scorer = scorer
        self.proposal = proposal

        N = init_board.shape[0]
        self.score = self.scorer.score(self.board)     # (N,)

        # Temperature Schduler        
        self.scheduler = scheduler

        # --- per-board state ---
        self.temperatures = torch.full((N,), temperature, device=self.board.device)
        self.steps = torch.full((N,), min_steps, device=self.board.device, dtype=torch.long)
        
        # Adaptive step settings
        self.adaptive_steps = adaptive_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.increase_factor = increase_factor
        self.patience = patience
        self.threshold = 0.01 
        self.max_increment = 256

        if adaptive_steps:
            self.scorer.steps = self.min_steps

        # keep small rolling windows per board
        self.step_history = [[] for _ in range(N)]  
        self.last_accept_rate = torch.zeros(N, device=self.board.device)    
        
    def step(self):
        """
        Perform one Metropolis-Hastings MCMC step for all boards in the batch.

        Updates each board with probability based on the change in score
        and optionally adjusts the number of simulation steps adaptively.

        Returns:
            score: torch.Tensor, current scores of each board in batch
            accept_mask: torch.BoolTensor, shape (N,), True if candidate accepted

        Safeguards:
            - Clamps probability for negative delta to avoid overflow.
            - Uses damping to prevent large adaptive step increases near max_steps.
            - Keeps step_history limited to `patience` entries.
        """
        candidate = self.proposal.propose(self.board)       # (N,H,W)
        candidate_score = self.scorer.score(candidate)      # (N,)

        delta = candidate_score - self.score                # (N,)
        
        # per-board acceptance
        rand = torch.rand_like(delta)
        accept_prob = torch.exp(torch.clamp(delta / self.temperatures, max=0))
        accept_mask = (delta >= 0) | (rand < accept_prob)

        # update accepted boards
        self.board._states[accept_mask] = candidate._states[accept_mask]
        self.score[accept_mask] = candidate_score[accept_mask]

        # update adaptive step length per board
        if self.adaptive_steps:
            for i in range(len(self.score)):
                self.step_history[i].append(self.score[i].item())
                if len(self.step_history[i]) > self.patience:
                    delta_avg = (self.step_history[i][-1] - self.step_history[i][0]) / self.patience
                    if delta_avg < self.threshold:
                        raw_increment = int(self.steps[i].item() * self.increase_factor)
                        damping = math.exp(- (self.steps[i].item() / self.max_steps) * 2)
                        increment = int(min(raw_increment * damping, self.max_increment))
                        if increment > 0:
                            self.steps[i] = min(self.steps[i] + increment, self.max_steps)
                    self.step_history[i].pop(0)

        # optional: temperature annealing per board
        if self.scheduler:
            self.temperatures = self.scheduler.get_temperature(len(self.step_history), prev_temps=self.temperatures)

        self.last_accept_rate = accept_mask.float()
        return self.score, accept_mask
    def get_results(self):
        """
        Retrieve current boards and their scores as numpy arrays.

        Returns:
            dict: {
                'boards': list of numpy arrays, each shape (H,W),
                'score': torch.Tensor of shape (N,) with current scores
            }

        Safeguards:
            - Ensures conversion to CPU numpy arrays.
            - Supports multiple chains in batch.
        """
        boards_np = [self.board.get_numpy(i) for i in range(self.board.n_chains())]
        return {
            "boards": boards_np,
            "score": self.score,
        }
