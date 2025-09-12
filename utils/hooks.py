import os
import time
from datetime import datetime

import torch

from utils.encodings import rle_encode_binary
from utils.logging import get_logger

def make_rle_saver(
    outdir: str,
    step_interval: int = None,
    time_interval: float = None,
    score_threshold: float = None,
    rule: str = "B3/S23"
):
    """
    Create a post-step hook that saves boards in RLE format
    with their scores and chain name.

    You can choose between step-based or time-based saving.

    Args:
        outdir: directory where RLE files are written
        step_interval: save every N sampler steps (optional)
        time_interval: save every Δt seconds (optional)
        score_threshold: only save if score >= threshold (optional)
        rule: GoL rule string for encoding
    """
    os.makedirs(outdir, exist_ok=True)
    last_save_time = {"t": time.time()}  # closure for time tracking

    def hook(t: int, chains: list, scores: torch.Tensor):
        logger = get_logger()
        # --- Decide if saving is triggered ---
        triggered = False
        if step_interval is not None and (t + 1) % step_interval == 0:
            triggered = True
        if time_interval is not None:
            now = time.time()
            if now - last_save_time["t"] >= time_interval:
                triggered = True
                last_save_time["t"] = now
        if not triggered:
            return

        # --- Save boards ---
        for idx, (chain, score) in enumerate(zip(chains, scores)):
            batch_size = chain.board.tensor.shape[0]
            for i in range(batch_size):
                board = chain.board.tensor[i]
                score_val = float(score[i].item())
                if score_threshold is not None and score_val < score_threshold:
                    continue  # skip uninteresting boards

                # Encode current board
                rle_str = rle_encode_binary(board.cpu(), rule=rule)

                # Make filename with timestamp, chain name, step & score
                chain_name = getattr(chain.scorer, "name", f"chain{idx}")
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                fname = f"{chain_name}_step{t+1}_score{int(score_val)}_{timestamp}.rle"

                # Write to file
                with open(os.path.join(outdir, fname), "w") as f:
                    f.write(f"# Chain: {chain_name}\n")
                    f.write(f"# Step: {t+1}\n")
                    f.write(f"# Score: {score_val:.4f}\n")
                    f.write(rle_str + "\n")
        msg = str(f"[Hooks] Saved {len(chains)*batch_size} model configurations as RLE files to {outdir}") 
        logger.debug(msg) if logger is not None else print(msg)


    return hook

def make_reheating_hook(
    min_accept_rate: float = 0.05,
    min_score_delta: float = None,
    lookback: int = 10,
    boost_factor: float = 1.5,
    max_temp: float = 10.0,
    verbose: bool = True,
):
    """
    Create a post-step hook that reheats *boards* if they stagnate.

    Triggers if:
      - acceptance rate for that board falls below `min_accept_rate`, OR
      - score improvement over `lookback` steps < `min_score_delta`.

    Args:
        min_accept_rate: minimum acceptable acceptance rate (float)
        min_score_delta: minimum score improvement over lookback window (optional)
        lookback: number of steps to consider for score stagnation
        boost_factor: factor to multiply board.temperature by
        max_temp: cap for maximum reheated temperature
        verbose: print messages when reheating
    """
    # Closure state: track score history per (chain_idx, board_idx)
    score_history = {}

    def hook(t: int, chains: list, scores: torch.Tensor):
        """
        Args:
            t: current step index
            chains: list of Chain objects (each has batch_size boards)
            scores: tensor [num_chains, batch_size] of scores
        """
        logger = get_logger()
        if t < 25:
            return
        for chain_idx, (chain, score_tensor) in enumerate(zip(chains, scores)):
            # assume chain.last_accept_rate is a tensor [batch_size]
            acc_rates = getattr(chain, "last_accept_rate", None)
            if acc_rates is None:
                continue

            batch_size = score_tensor.shape[0]
            for board_idx in range(batch_size):
                acc_rate = float(acc_rates[board_idx].item())
                score_val = float(score_tensor[board_idx].item())

                key = (chain_idx, board_idx)
                hist = score_history.setdefault(key, [])
                hist.append(score_val)
                if len(hist) > lookback:
                    hist.pop(0)

                # Check triggers
                trigger_acc = acc_rate < min_accept_rate
                trigger_score = (
                    min_score_delta is not None
                    and len(hist) == lookback
                    and (hist[-1] - hist[0]) < min_score_delta
                )

                if trigger_acc or trigger_score:
                    old_temp = float(chain.temperatures[board_idx].item())
                    new_temp = min(old_temp * boost_factor, max_temp)
                    chain.temperatures[board_idx] = new_temp

                    if verbose:
                        reason = []
                        if trigger_acc:
                            reason.append(f"low acc={acc_rate:.3f}")
                        if trigger_score:
                            reason.append(f"score={hist[-1]-hist[0]:.3f}")
                        msg = str(f"[Hooks] Reheating Step {t}, chain {chain_idx}, board {board_idx}: "
                            f"T {old_temp:.3f} -> {new_temp:.3f} ({', '.join(reason)})")
                        logger.debug(msg) if logger is not None else print(msg)

    return hook
