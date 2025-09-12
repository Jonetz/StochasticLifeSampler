import math
import random
from typing import Callable, List, Optional

import torch

from utils.logging import get_logger
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
                 cancel_condition: Optional[Callable[[int, List[Chain]], bool]] = None,
                 swap_interval: int = 50):   # new
        self.chains = chains
        
        # init logger
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None        
        self.logger = get_logger(log_dir=log_dir)

        self.pre_step_hooks = pre_step_hooks or []
        self.post_step_hooks = post_step_hooks or []
        self.cancel_condition = cancel_condition
        self.swap_interval = swap_interval

    def attempt_swaps(self, t: int):
        for i in range(len(self.chains) - 1):
            c1, c2 = self.chains[i], self.chains[i+1]

            e1 = -c1.score
            e2 = -c2.score
            T1 = c1.temperatures
            T2 = c2.temperatures

            dE = e2 - e1
            dT = (1.0 / T1 - 1.0 / T2)
            p_swap = torch.exp((dE * dT).clamp(max=50))  # avoid overflow
            p_swap = torch.minimum(p_swap, torch.ones_like(p_swap))

            # sample swap decisions
            mask = (torch.rand_like(p_swap) < p_swap)

            if mask.any():
                # swap states
                states1 = c1.board.tensor[mask].clone()
                states2 = c2.board.tensor[mask].clone()
                c1.board.tensor[mask], c2.board.tensor[mask] = states2, states1

                # swap scores
                s1 = c1.score[mask].clone()
                s2 = c2.score[mask].clone()
                c1.score[mask], c2.score[mask] = s2, s1

                # swap temps
                t1 = c1.temperatures[mask].clone()
                t2 = c2.temperatures[mask].clone()
                c1.temperatures[mask], c2.temperatures[mask] = t2, t1


                msg =  f"[Swap] Step {t}: chains {i} <-> {i+1}, {mask.sum().item()} boards swapped"
                self.logger.debug(msg) if hasattr(self, "logger") else print(msg)

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
                msg = f'[Sampler] Run cancelled at step {t} due to cancel condition.'
                self.logger.info(msg) if hasattr(self, 'logger') else print(msg)
                break

            # --- pre-step hooks ---
            for hook in self.pre_step_hooks:
                try:
                    hook(t, self.chains)
                except Exception as e:
                    msg = f'[Sampler] Error in pre step hook {hook}: {e}'
                    self.logger.error(msg) if hasattr(self, 'logger') else print(msg)

            scores = []
            accept_rates = []
            score = 0
            for idx, chain in enumerate(self.chains):
                try:
                    score, accepted = chain.step()
                except Exception as e:
                    msg = f'[Sampler] Error in chain step: {e}' + f'| Aborting chain step!'
                    self.logger.error(msg) if hasattr(self, 'logger') else print(msg)
                scores.append(score)
                accept_rates.append(chain.last_accept_rate)
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar(f"{idx}/score_mean", score.mean().item(), t)
                    self.writer.add_scalar(f"{idx}/score_max", score.max().item(), t)
                    self.writer.add_scalar(f"{idx}/score_min", score.min().item(), t)
                    self.writer.add_scalar(f"{idx}/accept_rate_mean", chain.last_accept_rate.mean().item(), t)
                    self.writer.add_scalar(f"{idx}/steps_mean", chain.steps.float().mean().item(), t)

            step_scores_tensor = torch.stack(scores)   # (num_chains, N)
            step_accepts_tensor = torch.stack(accept_rates)  # (num_chains, N)
            history.append(step_scores_tensor.cpu())

            # --- Attempt-Swap ---
            if (t + 1) % self.swap_interval == 0 and self.swap_interval > 0:
                self.attempt_swaps(t+1)

            # --- post-step hooks ---
            for hook in self.post_step_hooks:
                try:
                    hook(t, self.chains, step_scores_tensor)
                except Exception as e:
                    msg = f'Error in post step hook {hook}: {e}'
                    self.logger.error(msg) if hasattr(self, 'logger') else print(msg)
            # TensorBoard: aggregated logging
            if self.writer:
                self.writer.add_scalar("Step/avg_score", step_scores_tensor.mean().item(), t)
                self.writer.add_scalar("Step/max_score", step_scores_tensor.max().item(), t)
                self.writer.add_scalar("Step/min_score", step_scores_tensor.min().item(), t)
                self.writer.add_scalar("Step/avg_accept_rate", step_accepts_tensor.mean().item(), t)

            if (t + 1) % log_interval == 0 :
                msg = str(f"Step {t+1}: "
                    f"avg score = {step_scores_tensor.mean():.3f}, "
                    f"max score = {step_scores_tensor.max():.3f}, "
                    f"min score = {step_scores_tensor.min():.3f}, "
                    f"accept rate = {step_accepts_tensor.mean():.3f}")
                self.logger.info(msg) if hasattr(self, 'logger') else print(msg)
        if self.writer:
            self.writer.close()

        results = {
            idx: {
                "final_board": chain.board.clone(),
                "final_score": chain.score.clone(),
                "final_steps": chain.steps.clone(),
                "final_accept_rate": chain.last_accept_rate.clone()
            }
            for idx, chain in enumerate(self.chains)
        }
        return results, history