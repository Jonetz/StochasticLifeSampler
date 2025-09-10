
import cProfile
import os
import pstats
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging

from datetime import datetime

from core.Board import Board
from core.GOLEngine import GoLEngine

from core.Scorer import AliveCellCountScorer, ChangeRateScorer, ChaosScorer, EntropyScorer, OscillationScorer
from core.ComplexScorer import CombinedScorer, MethuselahScorer
from core.Proposal import CombinedProposal, SingleFlipProposal, BlockFlipProposal
from core.ComplexProposal import AreaTransformProposal, PatchNetProposal, PatternInsertProposal


from utils.hooks import make_reheating_hook, make_rle_saver

from mcmc.Chain import Chain
from mcmc.Sampler import Sampler
from mcmc.Scheduler import ExponentialScheduler, PlateauExponentialScheduler, OscillatingScheduler

from utils.visualization import plot_history

def main():

    device = "cuda"
    engine = GoLEngine(device=device)
    steps = 500
    box_size = (40,40)

    # Init boards
    boards = Board.from_shape(N=32, H=400, W=400, device=device, fill_prob=0.35, fill_shape=box_size)

    # Scorer
    scorer_steps = 32000
    #scorer = ChaosScorer(engine, steps=32000)
    #scorer = OscillationScorer(engine, steps=512)
    #scorer = AliveCellCountScorer(engine)
    scorer = MethuselahScorer(engine, steps=scorer_steps)
    #scorer = ChangeRateScorer(engine, steps=4096)
    #scorer = CombinedScorer(engine, 
    #    [(ChaosScorer(engine, steps=scorer_steps), 1),
    #     (MethuselahScorer(engine, steps=scorer_steps), 2),
    #     (ChangeRateScorer(engine, scorer_steps), 1)]
    #)

    #Scheduler
    #scheduler = OscillatingScheduler(start_temp=1.0, end_temp=0.2, steps=steps)
    scheduler = ExponentialScheduler(start_temp=2.0, end_temp=0.5, steps=steps)

    # Chains with different proposals
    def make_chain(proposal):
        return Chain(boards.clone(), scorer, proposal, scheduler=scheduler, adaptive_steps=True, max_steps=32000)
    chains = [
        #make_chain(SingleFlipProposal(use_activity_boundary=False, box_size=box_size)),
        #make_chain(BlockFlipProposal(box_size=box_size)),
        make_chain(CombinedProposal( [
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 25),
            (BlockFlipProposal(box_size=box_size), 75)])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 75),
            (BlockFlipProposal(box_size=box_size), 25)])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10)])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10)])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10)])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10)])),
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10)])),
        #make_chain(PatternInsertProposal(rle_folder=r"data\5x5",
        #                                max_files=500,
        #                                box_size=box_size,
        #                                target_shape=(5,5))),
        #make_chain(PatchNetProposal(filepath='data/network_final.pth',
        #                            box_size=box_size,
        #                            env_size=7,
        #                            patch_size=3))
    ]
    """
    chains = []
    for _ in range(10):
        chains.append(
        make_chain(CombinedProposal([
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 65),
            (BlockFlipProposal(box_size=box_size), 25),
            (AreaTransformProposal(box_size=box_size), 10)])))
    for _ in range(5):
        chains.append(make_chain(CombinedProposal( [
            (SingleFlipProposal(use_activity_boundary=False, box_size=box_size), 25),
            (BlockFlipProposal(box_size=box_size), 75)])))
    """

    # Hooks for peridoically dumping Boards:
    saver_hook = make_rle_saver(
        outdir=r"results\chain_samples",
        step_interval=100,          # save every 50 steps
        time_interval=60*15
    )
    reheat_hook = make_reheating_hook(min_accept_rate=0.05, min_score_delta=1.0)


    # Sampler
    log_folder = os.path.join(f'results', f'logs', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sampler = Sampler(chains, log_dir=log_folder, post_step_hooks=[saver_hook, reheat_hook])
    profiler = cProfile.Profile()
    profiler.enable()
    results, history = sampler.run(steps=steps, log_interval=1)
    folder = os.path.join('results', 'mcmc')
    profiler.disable()
    ps = pstats.Stats(profiler).sort_stats('cumtime')  # sort by cumulative time
    ps.print_stats(50)  # print top 50 lines
    os.makedirs(folder, exist_ok=True)
    for idx, r in results.items():
        final_board = r["final_board"]
        _, traj = engine.simulate(final_board, steps=5000, return_trajectory=True)
        
        # traj is a list of (1,H,W) or (B,H,W) tensors, convert to numpy
        traj_np = [b[0].cpu().numpy() if b.ndim == 3 else b.cpu().numpy() for b in traj]

        filename = os.path.join(folder, f"chain_{idx}_{int(r['final_score'].mean())}.gif")
        engine.trajectory_to_gif(traj_np, filename, fps=100)
    print(f'Save trajectories successfully to {folder}')
    plot_history(history, show_chains=True)


from utils.neural_proposals import main as training_main
if __name__ == "__main__":
    #training_main()
    main()