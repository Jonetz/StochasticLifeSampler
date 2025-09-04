import torch
from core.Board import Board
from core.Scorer import Scorer
from core.Proposal import Proposal

class Chain:
    """
    A single MCMC chain for a batched Game of Life board.
    """
    def __init__(self, 
                 init_board: Board, 
                 scorer: Scorer,
                 proposal: Proposal,
                 temperature: float = 1.0):
        self.board = init_board        # Board with shape (N,H,W)
        self.scorer = scorer
        self.proposal = proposal
        self.temperature = temperature
        self.score = self.scorer.score(self.board)  # shape (N,)

    def step(self):
        """
        Perform one MCMC step with MH acceptance per board in the batch.
        """
        candidate = self.proposal.propose(self.board)       # (N,H,W)
        candidate_score = self.scorer.score(candidate)      # (N,)

        delta = candidate_score - self.score                # (N,)

        # Acceptance probabilities per board
        accept_prob = torch.exp(torch.clamp(delta / self.temperature, max=0))  # (N,)
        # Clamp delta / T <= 0 for negative deltas, otherwise prob=1
        accept_mask = (delta >= 0) | (torch.rand_like(delta) < accept_prob)    # (N,)

        # Apply acceptance per board
        self.board._states[accept_mask] = candidate._states[accept_mask]
        self.score[accept_mask] = candidate_score[accept_mask]

        return self.score, accept_mask

    def get_results(self):
        """
        Retrieve the current board(s) and score(s) as numpy arrays.
        Returns:
            dict with keys:
                'boards': list of numpy arrays (H,W)
                'score': tensor or float of scores
        """
        boards_np = [self.board.get_numpy(i) for i in range(self.board.n_chains())]
        return {
            "boards": boards_np,
            "score": self.score,
        }
