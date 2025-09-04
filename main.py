
import os
import sys

from tqdm import tqdm

from core.ComplexProposal import PatternInsertProposal

# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from core.Board import Board
from core.GOLEngine import GoLEngine
import time
import cProfile, pstats
import numpy as np

"""
if __name__ == "__main__":
    
    profiler = cProfile.Profile()
    profiler.enable()

    
    # ---------------------------
    # Quick Demo Random
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    engine = GoLEngine(device=device, border="wrap")

    # Create 8 random chains of 30x30 with 10% fill
    board = engine.make_random(N=250, H=100, W=100, fill_prob=0.1)
    print("Initial batch shape:", board.shape)

    # Simulate WITH trajectory
    final_board, traj = engine.simulate(board, steps=1024, return_trajectory=True)
    print(f"Finished simulation!")


    # Save initial states RLE
    rle_path = "initial_states.json"
    engine.save_initial_states_rle(board, rle_path)
    print(f"Saved initial states RLE to {rle_path}")

    # Visualize chain 0 trajectory to GIF
    try:
        chain0_frames = [t[0] for t in traj]
        gif_path = "traj.gif"
        engine.trajectory_to_gif(chain0_frames, gif_path, fps=12, scale=4)
        print(f"Saved trajectory gif to {gif_path}")
    except Exception as e:
        print("Could not save gif (missing libs?):", e)

    # Show final live counts
    counts = final_board._states.sum(dim=(1, 2)).cpu().numpy()
    for i, c in enumerate(counts):
        print(f"Chain {i}: final live cells = {int(c)}")

    # ---------------------------
    # Glider
    # ---------------------------

    glider = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [0,0,1,1,1,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,1,1,1,0,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ], dtype=np.uint8)

    ship = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.uint8)

    # Place glider in a larger board (e.g. 10x10)
    H, W = 100, 100

    # Make batched Board with 1 chain
    engine = GoLEngine(device='cuda', border='constant')  # or 'cpu' if no GPU

    board = Board.from_numpy_list([ship], device='cuda', board_size=(H,W))
    # Simulate WITH trajectory
    final_board, traj = engine.simulate(board, steps=1000, return_trajectory=True)

    
    try:
        chain0_frames = [t[0] for t in traj]
        gif_path = "traj.gif"
        engine.trajectory_to_gif(chain0_frames, gif_path, fps=12, scale=4)
        print(f"Saved trajectory gif to {gif_path}")
    except Exception as e:
        print("Could not save gif (missing libs?):", e)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')  # Sort by cumulative time
    #stats.print_stats()  # Display the results

    
    """

from core.Board import Board
from core.GOLEngine import GoLEngine
from core.ComplexScorer import MethuselahScorer
from core.Proposal import CombinedProposal, SingleFlipProposal, BlockFlipProposal
from mcmc.Chain import Chain
from mcmc.Sampler import Sampler
from utils.visualization import plot_history

def main():
    device = "cuda"
    engine = GoLEngine(device=device)
    steps = 256
    box_size = (25,25)

    # Init boards
    boards = Board.from_shape(N=3, H=350, W=350, device=device, fill_prob=0.1, fill_shape=(10,10))

    # Scorer
    scorer = MethuselahScorer(engine, steps=1024)

    
    # Chains with different proposals
    chains = [
        Chain(boards.clone(), scorer, SingleFlipProposal(engine, use_activity_boundary=False, box_size=box_size)),
        Chain(boards.clone(), scorer, BlockFlipProposal(engine, box_size=box_size)),
        Chain(boards.clone(), scorer, CombinedProposal(engine, [
            (SingleFlipProposal(engine, use_activity_boundary=False, box_size=box_size), 25), 
            (BlockFlipProposal(engine, box_size=(25,25)), 75)])),
        Chain(boards.clone(), scorer, PatternInsertProposal(engine,
                                 rle_folder=r"C:\Users\jonas\Desktop\GOL\data\5x5",
                                 max_files=500,
                                 box_size=box_size,        # place patterns only inside central 32x32
                                 target_shape=(5,5),  # how RLEs are decoded (optional)
                                 device='cuda'))
    ]

    # Sampler
    sampler = Sampler(chains)
    results, history = sampler.run(steps=steps, log_interval=1)

    plot_history(history, show_chains=True)
    
    folder = os.path.join('results', 'mcmc')
    os.makedirs(folder, exist_ok=True)
    for idx, r in results.items():
        final_board = r["final_board"]
        _, traj = engine.simulate(final_board, steps=2048, return_trajectory=True)
        
        # traj is a list of (1,H,W) or (B,H,W) tensors, convert to numpy
        traj_np = [b[0].cpu().numpy() if b.ndim == 3 else b.cpu().numpy() for b in traj]

        filename = os.path.join(folder, f"chain_{idx}_{int(r['final_score'].mean())}.gif")
        engine.trajectory_to_gif(traj_np, filename)


if __name__ == "__main__":
    main()
