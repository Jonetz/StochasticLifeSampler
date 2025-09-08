
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime

from core.Board import Board
from core.GOLEngine import GoLEngine

from core.Scorer import AliveCellCountScorer, ChangeRateScorer, EntropyScorer, OscillationScorer
from core.ComplexScorer import MethuselahScorer
from core.Proposal import CombinedProposal, SingleFlipProposal, BlockFlipProposal
from core.ComplexProposal import AreaTransformProposal, PatchNetProposal, PatternInsertProposal

from utils.hooks import make_rle_saver

from mcmc.Chain import Chain
from mcmc.Sampler import Sampler

from utils.visualization import plot_history

def main():

    device = "cuda"
    engine = GoLEngine(device=device)
    steps = 2500
    box_size = (40,40)

    # Init boards
    boards = Board.from_shape(N=2, H=400, W=400, device=device, fill_prob=0.35, fill_shape=(40,40))

    # Scorer
    #scorer = EntropyScorer(engine, steps=512)
    #scorer = OscillationScorer(engine, steps=512)
    #scorer = AliveCellCountScorer(engine)
    scorer = MethuselahScorer(engine, steps=32000)
    #scorer = ChangeRateScorer(engine, steps=4096)

    # Chains with different proposals
    def make_chain(proposal):
        return Chain(boards.clone(), scorer, proposal, adaptive_steps=True, max_steps=32000)
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
        #make_chain(PatternInsertProposal(rle_folder=r"data\5x5",
        #                                max_files=500,
        #                                box_size=box_size,
        #                                target_shape=(5,5))),
        #make_chain(PatchNetProposal(filepath='data/network_final.pth',
        #                            box_size=box_size,
        #                            env_size=7,
        #                            patch_size=3))
    ]

    # Hooks for peridoically dumping Boards:
    saver_hook = make_rle_saver(
        outdir=r"results\chain_samples",
        step_interval=100,          # save every 50 steps
        time_interval=60*15
    )

    # Sampler
    log_folder = os.path.join(f'results', f'logs', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sampler = Sampler(chains, log_dir=log_folder, post_step_hooks=[saver_hook])
    results, history = sampler.run(steps=steps, log_interval=1)

    folder = os.path.join('results', 'mcmc')
    os.makedirs(folder, exist_ok=True)
    for idx, r in results.items():
        final_board = r["final_board"]
        _, traj = engine.simulate(final_board, steps=16000, return_trajectory=True)
        
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