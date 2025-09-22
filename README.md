<img src="data\GOL-MCMC.gif" alt="GOL-MCMC simulation" width="800">

# Stochastic Life Sampler: Exploring Conway's Game of Life with Stochastic Intelligence
Discover fascinating patterns in Conway's Game of Life using **Markov Chain Monte Carlo (MCMC)**! 

This GPU-accelerated framework goes beyond random searches, letting you optimize initial board configurations for methuselahs, oscillators, diehards, chaotic structures, and more.

---
## 🚀 Features

- **Fast GPU Simulation**: `GoLEngine` accelerates Game of Life simulations using PyTorch tensors.
- **Stochastic Search via MCMC**:
  - Chains with Metropolis-Hastings acceptance rules.
  - Flexible proposals: single flips, block flips, area transforms, pattern insertion, neural-network-based patches.
  - Temperature schedulers for balancing exploration and exploitation: Exponential, PlateauExponential, Oscillating, Adaptive.
- **Scoring Framework**:
  - Alive cell count, stability, change rate, entropy, chaos, diversity, oscillation, methuselah scoring.
  - Combine multiple scorers for custom objectives.
- **Sampler Orchestration**:
  - Multi-chain management.
  - Hooks for saving states, adaptive reheating, and logging.
- **Visualization & Logging**:
  - Render boards to images, GIFs, or videos.
  - Experiment tracking and performance plots.
- **Customizability**:
  - The framework is build to a low threshold of customizability, no component is fixed!
  - Hooks, Scorers, Simulations etc. are build to be efficiently replaced using custom components.

---

## ⚡ Installation

```bash
git clone https://github.com/Jonetz/GOL.git
cd GOL
pip install -r requirements.txt
```

> Requires Python 3.9+ and PyTorch with CUDA for GPU acceleration.

---

## 🧩 Quickstart Example

```python
from GOL.core.Board import Board
from GOL.core.GoLEngine import GoLEngine
from GOL.core.Scorer import CombinedScorer, ChaosScorer, MethuselahScorer, ChangeRateScorer
from GOL.mcmc.Chain import Chain
from GOL.mcmc.Sampler import Sampler
from GOL.mcmc.Scheduler import ExponentialScheduler
from GOL.core.Proposal import SingleFlipProposal, BlockFlipProposal, AreaTransformProposal, NewBoardProposal, CombinedProposal
import os
from datetime import datetime

# Device setup
device = 'cuda'
engine = GoLEngine(device=device)
steps = 2500
box_size = (16,16)

# Initialize boards
boards = Board.from_shape(N=16, H=400, W=400, device=device, fill_prob=0.35, fill_shape=box_size)

# Define scorer
scorer_steps = 55000
scorer = CombinedScorer(engine, [
    (ChaosScorer(engine, steps=scorer_steps), 1),
    (MethuselahScorer(engine, steps=scorer_steps), 1),
    (ChangeRateScorer(engine, scorer_steps), 1)
])

# Scheduler
scheduler = ExponentialScheduler(start_temp=1.0, end_temp=0.2, steps=steps)

# Define chains
def make_chain(proposal):
    return Chain(boards.clone(), scorer, proposal, scheduler=scheduler, adaptive_steps=True, max_steps=20000)

chains = [
    make_chain(CombinedProposal([
        (SingleFlipProposal(box_size=box_size), 60),
        (BlockFlipProposal(box_size=box_size), 20),
        (AreaTransformProposal(box_size=box_size), 10),
        (NewBoardProposal(box_size=box_size), 10)
    ]))
]

# Sampler and logging
log_folder = os.path.join('results', 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
sampler = Sampler(chains, log_dir=log_folder)
results, history = sampler.run(steps=steps, log_interval=1)
```

---

## 📂 Repository Structure

```
GOL/
├── core/
│   ├── Board.py           # Board representation and initialization
│   ├── GoLEngine.py       # GPU-accelerated Game of Life engine
│   ├── Scorer.py          # Scoring functions and combined scorers
│   └── Proposal.py        # Proposal strategies for MCMC
├── mcmc/
│   ├── Chain.py           # Single MCMC chain management
│   ├── Sampler.py         # Multi-chain orchestration and experiment loop
│   └── Scheduler.py       # Temperature scheduling strategies
├── utils/
│   ├── Visualization.py   # Render boards and GIFs/videos
│   ├── Storage.py         # Save/load board states
│   └── Logging.py         # Experiment logging utilities
└── main.py                # CLI / experiment runner
```

---

## 📚 References
- [Describing Post](https://jonetz.github.io/year-archive/) 
  
- [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway's_Game_of_Life)
- [Metropolis-Hastings Algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- [APGSearch](https://conwaylife.com/wiki/Apgsearch)

---

## ⚖️ License

MIT License. See `LICENSE` for details.