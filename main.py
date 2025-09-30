
import cProfile
import os
import pstats
import sys

import numpy as np

from utils.encodings import rle_decode_binary

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging

from datetime import datetime

from core.Board import Board
from core.GOLEngine import GoLEngine

from core.Scorer import AliveCellCountScorer, ChangeRateScorer, ChaosScorer, EntropyScorer, OscillationScorer
from core.ComplexScorer import CombinedScorer, CompactnessScorer, MethuselahScorer, MovementScorer
from core.Proposal import CombinedProposal, SingleFlipProposal, BlockFlipProposal
from core.ComplexProposal import AreaTransformProposal, PatchNetProposal, PatternInsertProposal, NewBoardProposal


from utils.hooks import make_reheating_hook, make_rle_saver

from mcmc.Chain import Chain
from mcmc.Sampler import Sampler
from mcmc.Scheduler import ExponentialScheduler, PlateauExponentialScheduler, OscillatingScheduler

from utils.visualization import plot_history

def main():

    device = "cuda"
    engine = GoLEngine(device=device)
    steps = 250
    box_size = (16,16)

    # Init boards
    boards = Board.from_shape(N=16, H=800, W=800, device=device, fill_prob=0.35, fill_shape=box_size)

    # Scorer
    scorer_steps = 55000
    #scorer = ChaosScorer(engine, steps=32000)
    #scorer = OscillationScorer(engine, steps=512)
    #scorer = AliveCellCountScorer(engine)
    scorer = MethuselahScorer(engine, steps=scorer_steps)
    #scorer = ChangeRateScorer(engine, steps=4096)
    scorer = CombinedScorer(engine, 
        [(ChaosScorer(engine, steps=scorer_steps), 0.5),
         (MethuselahScorer(engine, steps=scorer_steps), 10),
         (ChangeRateScorer(engine, scorer_steps), 1)]
    )

    #Scheduler
    #scheduler = OscillatingScheduler(start_temp=1.0, end_temp=0.2, steps=steps)
    scheduler = PlateauExponentialScheduler(start_temp=1.0, end_temp=0.2, steps=steps)
    #scheduler = ExponentialScheduler(start_temp=1.0, end_temp=0.2, steps=steps)

    # Chains with different proposals
    def make_chain(proposal):
        return Chain(boards.clone(), scorer, proposal, scheduler=scheduler, adaptive_steps=True, max_steps=20000)
    chains = [
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 75),
            (BlockFlipProposal(box_size=box_size), 25)
            (AreaTransformProposal(box_size=box_size), 10),
            (NewBoardProposal(fill_prob=0.35, box_size=box_size), 10)            ])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10),
            (NewBoardProposal(fill_prob=0.35, box_size=box_size), 10)            ])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10),
            (NewBoardProposal(fill_prob=0.35, box_size=box_size), 10)            ])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10),
            (NewBoardProposal(fill_prob=0.35, box_size=box_size), 10)            ])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10),
            (NewBoardProposal(fill_prob=0.35, box_size=box_size), 10)            ])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10),
            (NewBoardProposal(fill_prob=0.35, box_size=box_size), 10)            ]))
    ]

    # Sampler
    log_folder = os.path.join(f'results', f'logs', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Hooks for peridoically dumping Boards & Reheating stale chains:
    saver_hook = make_rle_saver(
        outdir=os.path.join(log_folder, r"chain_samples"),
        step_interval=100,          # save every 50 steps
        time_interval=60*15
    )
    reheat_hook = make_reheating_hook(min_accept_rate=0.05, min_score_delta=1.0, boost_factor=5)


    sampler = Sampler(chains, log_dir=log_folder, post_step_hooks=[saver_hook, reheat_hook])
    
    results, history = sampler.run(steps=steps, log_interval=1)
    
    #plot_history(history, show_chains=True)


from utils.neural_proposals import main as training_main
if __name__ == "__main__":
    #training_main()
    while True:
        main()