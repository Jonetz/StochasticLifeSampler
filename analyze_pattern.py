import os
import shutil
import sys
from typing import Union

from core.ComplexScorer import MethuselahScorer, RotInvariantScorer
from utils.encodings import load_rle_files, rle_decode_binary

# Add the root project directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.Board import Board
from core.GOLEngine import GoLEngine
from core.Scorer import AliveCellCountScorer, ChangeRateScorer, DiversityScorer, EntropyScorer, OscillationScorer, StabilityScorer #, CompositScorer

import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# ---------- RLE PARSER ----------
def parse_rle(rle_text: str, target_shape: tuple[int,int] = None) -> np.ndarray:
    """
    Parse a ConwayLife RLE string into a numpy array of shape (H, W).
    Returns binary array (1=alive, 0=dead).

    target_shape: optional (H_max, W_max) to pad all arrays to the same size.
    """
    lines = [l.strip() for l in rle_text.splitlines() if not l.startswith("#")]
    header = lines[0]
    m = re.search(r"x\s*=\s*(\d+),\s*y\s*=\s*(\d+)", header)
    if not m:
        raise ValueError("RLE header missing x,y size")
    W, H = int(m.group(1)), int(m.group(2))
    if W > target_shape[0] or H > target_shape[1]:
        raise ValueError(f"Skipping due to large size (W={W}, H={H})")

    body = "".join(lines[1:])
    arr = np.zeros((H, W), dtype=np.uint8)

    x = y = 0
    tokens = re.findall(r"\d*[bo$!]", body)
    for tok in tokens:
        n = int(tok[:-1]) if tok[:-1].isdigit() else 1
        sym = tok[-1]
        if sym == "b":
            x += n
        elif sym == "o":
            for i in range(n):
                if 0 <= y < H and 0 <= x+i < W:
                    arr[y, x+i] = 1
            x += n
        elif sym == "$":
            y += n
            x = 0
        elif sym == "!":
            break

    # Optional: pad/cast to target_shape
    if target_shape is not None:
        H_max, W_max = target_shape
        if H_max < H or W_max < W:
            raise ValueError(f"Target shape {target_shape} is smaller than RLE array {(H,W)}")
        padded = np.zeros(target_shape, dtype=np.uint8)
        
        W, H = arr.shape
        offset_w = (H_max - W) // 2
        offset_h = (W_max  - H) // 2

            # copy array into board
        padded[offset_h:offset_h+H, offset_w:offset_w+W] = torch.from_numpy(arr.astype(np.uint8))
        return padded

    return arr



def plot_histograms(sizes, alives):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(sizes, bins=50)
    plt.xlabel("Total size (H*W)")
    plt.ylabel("Count")
    plt.title("Distribution of total pattern sizes")

    plt.subplot(1,2,2)
    plt.hist(alives, bins=50)
    plt.xlabel("Number of alive cells")
    plt.ylabel("Count")
    plt.title("Distribution of live cells")
    plt.tight_layout()
    plt.show()

# ---------- SIMULATION ----------
def simulate_pattern(arr_list, steps=128, engine=None, board_size=(100, 100)):
    """
    Simulate multiple patterns in a batch.
    arr_list: list of np.ndarray patterns
    returns: list of trajectories, one per pattern
    """
    if engine is None:
        engine = GoLEngine(device='cuda', border='constant')

    N = len(arr_list)
    # Create a batched Board tensor (N, H, W)
    boards = Board.from_numpy_list(arr_list, device='cuda', board_size=board_size)
    final_board, traj = engine.simulate(boards, steps=steps, return_trajectory=True)

    # Split trajectory into per-pattern lists
    batch_trajectories = []
    for n in range(N):
        batch_trajectories.append([step[n].cpu() for step in traj])

    return batch_trajectories

def rank(arrs, scorer, engine):
    best_patterns = {}
    simulation_time = 0.0
    scoring_times = {name: 0.0 for name in scorers}

    # ---------------- BATCH PREP ----------------
    boards = Board.from_numpy_list(arrs, device='cuda')  # (batch_size, H, W)

    # ---------------- SIMULATION ----------------
    start_simulation = time.time()
    #print("Beginning batched simulation")
    final_board = engine.simulate(boards, steps=1024)
    final_board = final_board.to('cpu')
    #final_board, traj = engine.simulate(boards, steps=1024, return_trajectory=True)
    #final_board, traj = final_board.to('cpu'), [pattern.to('cpu') for pattern in traj]
    #print("Ended simulation")
    simulation_time += time.time() - start_simulation

    # ---------------- SCORING ----------------
    for name, scorer in scorers.items():
        start_scoring = time.time()
        # Each scorer now receives the full batch
        if name == "Methuselah" or name == "RotInvariant":
            scores = scorer.score(boards)
        else:
            scores = scorer.score(final_board)  # returns tensor or np.array of shape (batch_size,)
        elapsed = time.time() - start_scoring
        scoring_times[name] += elapsed

        # Track best per scorer
        best_idx = int(scores.argmax())
        #best_traj = [step[best_idx] for step in traj]
        best_patterns[name] = (float(scores[best_idx]), arrs[best_idx])#, best_traj)
        
    # ---------------- RESULTS ----------------
    #print(f"Total simulation time: {simulation_time:.2f}s")
    #for name, t in scoring_times.items():
    #    print(f"Scorer '{name}': total scoring time = {t:.2f}s")

    return best_patterns

def save_results(results, engine, base_folder="results"):
    os.makedirs(base_folder, exist_ok=True)

    for scorer_name, scorer_results in results.items():
        print(f'Starting to save results for {scorer_name}...')
        scorer_folder = os.path.join(base_folder, scorer_name)
        os.makedirs(scorer_folder, exist_ok=True)

        # unpack scores, patterns, trajectories
        scores = [r[0] for r in scorer_results]
        patterns = [r[1] for r in scorer_results]
        #trajs = [r[2] for r in scorer_results]

        # sort by score (ascending = worst first)
        sorted_idx = np.argsort(scores)

        # pick three worst, three best, and one middle
        worst_indices = sorted_idx[:3]
        best_indices = sorted_idx[-3:]
        middle_index = sorted_idx[len(sorted_idx) // 2 : len(sorted_idx) // 2 + 1]

        selected_indices = list(worst_indices) + list(middle_index) + list(best_indices)

        engine = GoLEngine('cuda', border='constant')
        for rank_idx, i in enumerate(selected_indices):
            score = scores[i]
            pattern = patterns[i]
            _, traj = engine.simulate(pattern, 2048, return_trajectory=True)#trajs[i]

            print(f'Saving to {rank_idx:02d}_score_{score:.4f}')
            subfolder = os.path.join(scorer_folder, f"{rank_idx:02d}_score_{score:.4f}")
            os.makedirs(subfolder, exist_ok=True)

            # save pattern numpy
            engine.save_initial_states_rle(pattern, os.path.join(subfolder, "pattern.rle"))

            # save trajectory gif
            engine.trajectory_to_gif(traj, os.path.join(subfolder, "trajectory.gif"), fps=50)

# ---------- MAIN ----------
if __name__ == "__main__":
    folder = r"data\all"
    arrs, _, _, names = load_rle_files(folder, target_shape=(150,150))    
    scorers_steps = 64

    engine = GoLEngine(device='cuda', border='constant')

    """
    for name, arr in zip(names, arrs):
        print(name)                
        engine.show(board=torch.tensor(arr))
        shutil.copy(os.path.join(folder, name), os.path.join(r'data\5x5', name))
    exit()
    """

    scorers = {
        "AliveCellCount": AliveCellCountScorer(),
        "Stability": StabilityScorer(engine, scorers_steps),
        "ChangeRate": ChangeRateScorer(engine, scorers_steps),
        "Entropy": EntropyScorer(engine, scorers_steps),
        "Diversity": DiversityScorer(),
        "Oscillation": OscillationScorer(engine, scorers_steps),
        #"Composite": CompositeScorer(),
        "Methuselah": MethuselahScorer(engine, scorers_steps),
        "RotInvariant": RotInvariantScorer(engine, scorers_steps)
    }

    results = {name: [] for name in scorers}

    batch_size = 10
    for start in tqdm(range(0, len(arrs), batch_size)):
        batch = arrs[start:start+batch_size]
        ranked = rank(batch, scorers, engine)
        for name, r in ranked.items():
            results[name].append(r)


    save_results(results, engine)

    # ---------------- DISPLAY BEST ----------------
    for name, scorer_results in results.items():
        # take the highest scoring trajectory
        best_idx = np.argmax([r[0] for r in scorer_results])
        _, pattern = scorer_results[best_idx]
        _, traj = engine.simulate(pattern, 2048, return_trajectory=True)
        engine.trajectory_to_gif(traj, f'best_{name}.gif', fps=50)
