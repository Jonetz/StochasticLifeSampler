import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


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
