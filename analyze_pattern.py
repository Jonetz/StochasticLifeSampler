import os
import sys

# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.GOLEngine import GoLEngine
from utils.encodings import load_rle_files, rle_decode_binary

# ---------- MAIN ----------
if __name__ == "__main__":

    engine = GoLEngine(device='cuda', border='constant')
    fn = r'results\chain_samples\MethuselahScorer_step800_score3430.500_20250908-125341.rle'
    with open(fn, "r") as f:
        rle_str = f.read()
    print(rle_str)
    arr = rle_decode_binary(rle_str, device='cuda', target_shape=(250,250))
    print(arr)
    arr = arr.to('cpu').to(torch.uint8)
    engine.show(board=arr)
    _, traj = engine.simulate(arr, steps=4096, return_trajectory=True)
    gif_fn = fn.replace('.rle', '.gif')
    engine.trajectory_to_gif(traj, filepath=gif_fn, fps=50)
    exit()