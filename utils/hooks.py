import os
import time
from datetime import datetime

import torch

from utils.encodings import rle_encode_binary

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
                score_val = float(score.mean().item())
                if score_threshold is not None and score_val < score_threshold:
                    continue  # skip uninteresting boards

                # Encode current board
                rle_str = rle_encode_binary(board.cpu(), rule=rule)

                # Make filename with timestamp, chain name, step & score
                chain_name = getattr(chain.scorer, "name", f"chain{idx}")
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                fname = f"{chain_name}_step{t+1}_score{score_val:.3f}_{timestamp}.rle"

                # Write to file
                with open(os.path.join(outdir, fname), "w") as f:
                    f.write(f"# Chain: {chain_name}\n")
                    f.write(f"# Step: {t+1}\n")
                    f.write(f"# Score: {score_val:.4f}\n")
                    f.write(rle_str + "\n")

    return hook
