from typing import List

import torch
from .Chain import Chain

class Sampler:
    """
    Manages multiple chains in parallel.
    """
    def __init__(self, chains: List[Chain]):
        self.chains = chains

    def run(self, steps: int, log_interval: int = 100):
        history = []
        for t in range(steps):
            scores = []
            for chain in self.chains:
                score, accepted = chain.step()
                scores.append(score)
            step_scores_tensor = torch.stack(scores).float()
            # History to cpu to free gpu memory
            history.append(step_scores_tensor.to('cpu'))

            if (t+1) % log_interval == 0:
                print(f"Step {t+1}: avg score = {torch.stack(scores).float().mean():.3f}")
        
        results = {}
        for idx, chain in enumerate(self.chains):
            results[idx] = {
                "final_board": chain.board.clone(),
                "final_score": chain.score.clone()
            }
        return results, history
