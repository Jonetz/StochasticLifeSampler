from typing import Callable, List, Optional

import torch
from .Chain import Chain

from torch.utils.tensorboard import SummaryWriter

class Sampler:
    """
    Manages multiple MCMC chains in parallel.

    Each chain is updated independently per step, with optional hooks
    and TensorBoard logging support.

    Args:
        chains: list of Chain objects to run in parallel
        log_dir: optional directory for TensorBoard logs
        pre_step_hooks: list of callables, each called before a step:
                        hook(step_idx, chains)
        post_step_hooks: list of callables, each called after a step:
                        hook(step_idx, chains, scores_tensor)
        cancel_condition: callable(step_idx, chains) -> bool, allows early termination
    """
    def __init__(self, 
                 chains: List[Chain], 
                 log_dir: Optional[str] = None,
                 pre_step_hooks: Optional[List[Callable]] = None,
                 post_step_hooks: Optional[List[Callable]] = None,
                 cancel_condition: Optional[Callable[[int, List[Chain]], bool]] = None):
        self.chains = chains
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

        self.pre_step_hooks = pre_step_hooks or []
        self.post_step_hooks = post_step_hooks or []
        self.cancel_condition = cancel_condition


    def run(self, steps: int, log_interval: int = 100):
        """
        Run all chains for a given number of steps.

        Performs Metropolis-Hastings updates on each chain, applies hooks,
        logs to TensorBoard if enabled, and aggregates history.

        Args:
            steps: total number of steps to run
            log_interval: interval at which to print progress to stdout

        Returns:
            results: dict mapping chain index to dict with keys:
                'final_board': Board object after last step
                'final_score': tensor of scores for each board in batch
            history: list of torch.Tensor of scores per step, shape (num_chains, N)

        Safeguards:
            - Checks cancel_condition each step for early exit.
            - Ensures TensorBoard writer is closed properly.
            - Converts scores to CPU for history storage to avoid GPU memory bloat.
            - Handles multiple chains and batch sizes consistently.
        """
        history = []
        for t in range(steps):
            # --- cancel condition ---
            if self.cancel_condition and self.cancel_condition(t, self.chains):
                print(f"Run cancelled at step {t}")
                break

            # --- pre-step hooks ---
            for hook in self.pre_step_hooks:
                hook(t, self.chains)

            scores = []
            for idx, chain in enumerate(self.chains):
                score, accepted = chain.step()
                scores.append(score)
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar(f"{idx}/score", score.mean(), t)
                    self.writer.add_scalar(f"{idx}/steps", chain.scorer.steps, t)

            step_scores_tensor = torch.stack(scores).float()
            history.append(step_scores_tensor.to('cpu'))

            # --- post-step hooks ---
            for hook in self.post_step_hooks:
                hook(t, self.chains, step_scores_tensor)

            # TensorBoard: aggregated logging
            if self.writer:
                self.writer.add_scalar("Step/avg_score", step_scores_tensor.mean().item(), t)
                self.writer.add_scalar("Step/max_score", step_scores_tensor.max().item(), t)
                self.writer.add_scalar("Step/min_score", step_scores_tensor.min().item(), t)

            if (t+1) % log_interval == 0:
                print(f"Step {t+1}: avg score = {step_scores_tensor.mean():.3f}")

        if self.writer:
            self.writer.close()

        results = {
            idx: {"final_board": chain.board.clone(),
                  "final_score": chain.score.clone()}
            for idx, chain in enumerate(self.chains)
        }
        return results, history
