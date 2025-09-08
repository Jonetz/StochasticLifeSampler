import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_history(history, sigma: int = 2, show_chains: bool = False):
    """
    Plot MCMC score history.
    
    By default plots mean ± min/max envelope.
    Optionally plot all chains individually.

    Args:
        history: list of torch.Tensor, each (batch_size,)
                 scores at each step
        sigma: stddev for Gaussian smoothing
        show_chains: if True, plot each chain separately
    """
    steps = len(history)
    batch_size = history[0].shape[0]

    # Stack into numpy (steps, batch_size)
    arr = torch.stack(history).cpu().numpy()  # shape (steps, batch_size)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)   # (steps, batch_size)

    step_arr = np.arange(steps)
    if show_chains:
        plt.figure(figsize=(10, 5))
        for j in range(batch_size):
            smooth_chain = gaussian_filter1d(arr[:, j], sigma=sigma)
            plt.plot(step_arr, smooth_chain, alpha=0.4, label=f"Chain {j}" if j < 10 else None)
        plt.title("MCMC Score Evolution per Chain (Gaussian Smoothed)")
        plt.xlabel("Step")
        plt.ylabel("Score")
        if batch_size <= 10:  # avoid legend spam
            plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        means = arr.mean(axis=1)
        mins = arr.min(axis=1)
        maxs = arr.max(axis=1)

        smooth_means = gaussian_filter1d(means, sigma=sigma)
        smooth_mins = gaussian_filter1d(mins, sigma=sigma)
        smooth_maxs = gaussian_filter1d(maxs, sigma=sigma)

        plt.figure(figsize=(10, 5))
        plt.plot(step_arr, smooth_means, color='blue', label='Mean')
        plt.fill_between(step_arr, smooth_mins, smooth_maxs, 
                         color='lightblue', alpha=0.3, label='Min/Max')
        plt.title("MCMC Score Evolution (Gaussian Smoothed)")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Regex for RLE header and body tokens
_HEADER_RE = re.compile(r"x\s*=\s*(\d+),\s*y\s*=\s*(\d+)")
_TOKENS_RE = re.compile(r"(\d*)([boxy$!])", re.IGNORECASE)

def analyze_rle_folder_light(folder: str, max_files=None):
    """
    Efficiently analyze RLE files: extract height, width, alive cells.
    Plots height/width histograms and scatter plot, using logarithmic scale.
    """
    def remove_outliers(heights, widths, alive_counts, percentile=99):
        """
        Remove samples where height or width exceeds the given percentile.
        percentile: e.g., 99 means remove the top 1% largest values.
        Returns filtered heights, widths, alive_counts.
        """
        # Compute cutoff thresholds
        h_cut = np.percentile(heights, percentile)
        w_cut = np.percentile(widths, percentile)

        # Keep only samples below both cutoffs
        mask = (heights <= h_cut) & (widths <= w_cut)

        return heights[mask], widths[mask], alive_counts[mask]

    heights, widths, alive_counts = [], [], []

    files = [f for f in os.listdir(folder) if f.endswith(".rle")]
    if max_files:
        files = files[:max_files]

    for fname in files:
        path = os.path.join(folder, fname)
        try:
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f if not ln.strip().startswith("#")]

            # Parse header
            header = None
            for ln in lines:
                m = _HEADER_RE.search(ln)
                if m:
                    header = m
                    break
            if header is None:
                print(f"Skipped {fname}: missing header")
                continue

            W, H = int(header.group(1)), int(header.group(2))
            widths.append(W)
            heights.append(H)

            # Count alive cells in body
            idx = lines.index(header.string)
            body = "".join(lines[idx + 1:])
            alive = 0
            x = y = 0
            for count_str, tag in _TOKENS_RE.findall(body):
                count = int(count_str) if count_str else 1
                tag = tag.lower()
                if tag in ("b",):
                    x += count
                elif tag in ("o", "x", "y", "z"):  # alive
                    alive += count
                    x += count
                elif tag == "$":
                    y += count
                    x = 0
                elif tag == "!":
                    break
                else:
                    x += count
            alive_counts.append(alive)

        except Exception as e:
            print(f"Skipped {fname}: {e}")

    heights = np.array(heights)
    widths = np.array(widths)
    alive_counts = np.array(alive_counts)
    heights, widths, alive_counts = remove_outliers(heights, widths, alive_counts, 99.9)


    # Print summary
    print(f"Total files analyzed: {len(heights)}")
    print(f"Height: min={heights.min()}, max={heights.max()}, mean={heights.mean():.2f}")
    print(f"Width: min={widths.min()}, max={widths.max()}, mean={widths.mean():.2f}")
    print(f"Alive cells: min={alive_counts.min()}, max={alive_counts.max()}, mean={alive_counts.mean():.2f}")


    # Plot histograms and scatter
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(heights, bins=50, log=True, color='skyblue', edgecolor='black')
    plt.xlabel("Height")
    plt.ylabel("Count (log scale)")
    plt.title("Height distribution")

    plt.subplot(1, 3, 2)
    plt.hist(widths, bins=50, log=True, color='salmon', edgecolor='black')
    plt.xlabel("Width")
    plt.ylabel("Count (log scale)")
    plt.title("Width distribution")

    plt.subplot(1, 3, 3)
    plt.scatter(widths, heights, alpha=0.5, s=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Width vs Height scatter (log-log)")

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    analyze_rle_folder_light(r'data\all')