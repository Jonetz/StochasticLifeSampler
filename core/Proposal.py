import torch
from abc import abstractmethod
from typing import Optional, Tuple
from .Board import Board

class Proposal():
    """
    Base class for all proposal distributions in MCMC/board updates.

    Args:
        name: Optional name for the proposal.
        box_size: Optional bounding box (H_box, W_box) restricting proposals.
        device: Device to perform operations on (defaults to 'cuda' if available).

    Methods:
        propose(board): Abstract method to generate a proposed board state.

    Safeguards:
        - Subclasses must implement `propose`.
    """
    def __init__(self, name: Optional[str] = None, box_size: Optional[Tuple[int,int]] = None, device: Optional[str] = None):        
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.box_size = box_size  # (H_box, W_box)
        self.name = name or self.__class__.__name__

    @abstractmethod
    def propose(self, board: Board) -> Board:
        """
        Given current board state, return a proposed new board.
        """
        pass

class CombinedProposal(Proposal):
    """
    Combines multiple child proposals with specified probabilities.

    Each step, one child proposal is selected independently for each board
    according to its weight.

    Args:
        proposals: List of tuples (Proposal, weight).
        name: Optional name for the combined proposal.
        box_size: Optional bounding box to pass to child proposals.

    Methods:
        propose(board): Applies sampled proposal to each board in batch.

    Safeguards:
        - Normalizes weights to sum to 1.
        - Clones input board to avoid in-place modification.
        - Ensures proper indexing of batch and proposal.
    """
    def __init__(self, proposals: list[tuple[Proposal, float]], name: Optional[str] = None, box_size: Optional[Tuple[int,int]] = None):
        super().__init__(name)
        self.proposals = [p for p, _ in proposals]
        weights = torch.tensor([w for _, w in proposals if not w <= 0], dtype=torch.float32)
        self.probs = weights / weights.sum()  # normalize to sum=1
        self.box_size = box_size

    
    def propose(self, board: Board) -> Board:
        b = board.clone()
        N, H, W = b.shape

        # Sample a proposal index for each chain independently
        proposal_indices = torch.multinomial(self.probs, num_samples=N, replacement=True)

        # Apply each proposal to the corresponding chain
        for idx, prop_idx in enumerate(proposal_indices):
            prop = self.proposals[prop_idx]
            single_board = Board(b._states[idx:idx+1])  # view for this chain
            updated = prop.propose(single_board)
            b._states[idx] = updated._states[0]  # copy back

        return b

class SingleFlipProposal(Proposal):
    """
    Propose a new board by flipping a single random cell.

    Supports optional restriction to a central bounding box or
    to an activity boundary (cells that are changing).

    Args:
        use_activity_boundary: If True, only flip cells that are active.
        box_size: Optional central box (H_box, W_box) to restrict flips.

    Methods:
        propose(board): Returns a new board with one flipped cell per batch.

    Safeguards:
        - Uses uniform sampling if activity boundary not implemented.
        - Ensures indices are within bounds.
        - Clones the board to prevent modifying input.
        - Raises NotImplementedError if use_activity_boundary=True.
    """
    def __init__(self, use_activity_boundary=False, box_size=None):
        super().__init__(box_size=box_size)
        self.use_activity_boundary = use_activity_boundary
    def propose(self, board: Board) -> Board:
        b = board.clone()
        N, H, W = b.shape
        device = b._states.device

        # Determine bounding box
        if self.box_size is not None:
            H_box, W_box = self.box_size
            start_i = (H - H_box) // 2
            start_j = (W - W_box) // 2
            end_i = start_i + H_box
            end_j = start_j + W_box
        else:
            start_i, start_j, end_i, end_j = 0, 0, H, W

        if self.use_activity_boundary:
            raise NotImplementedError('This part is not working yet, hope to fix it in the future')
            # Compute activity mask
            next_state = self.engine.simulate(b.tensor)  # (N,H,W)
            activity_mask = (next_state != b.tensor)

            # Apply bounding box mask
            box_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
            box_mask[start_i:end_i, start_j:end_j] = 1
            activity_mask &= box_mask.unsqueeze(0)  # broadcast to (N,H,W)

            # Fallback for empty activity# Flatten H and W dimensions
            no_activity = ~activity_mask.any(dim=1)  # (N,)

            if no_activity.any():
                i_indices = torch.randint(start_i, end_i, (N,), device=device)
                j_indices = torch.randint(start_j, end_j, (N,), device=device)
            # Flatten and sample one active cell per batch
            flat_mask = activity_mask.view(N, -1)        # (N, H*W)
            # Get indices of all active cells
            active_positions = flat_mask.nonzero(as_tuple=False)  # (total_active, 2)
            # For each batch, pick a random one
            choices = torch.zeros(N, dtype=torch.long, device=device)
            for n in range(N):
                rows = active_positions[:,0] == n
                indices_n = active_positions[rows, 1]
                choices[n] = indices_n[torch.randint(0, len(indices_n), (1,), device=device)]

            i_indices = choices // W
            j_indices = choices % W

        else:
            # Uniform random within bounding box
            i_indices = torch.randint(start_i, end_i, (N,), device=device)
            j_indices = torch.randint(start_j, end_j, (N,), device=device)

        # Apply flips
        b._states[torch.arange(N, device=device), i_indices, j_indices] ^= 1
        return b

class BlockFlipProposal(Proposal):
    """
    Flip a 2x2 block of cells inside an optional central bounding box.

    Args:
        box_size: Optional central box (H_box, W_box) to restrict where blocks
                  are chosen. Must be large enough to fit 2x2 block.
        name: Optional name for the proposal.

    Methods:
        propose(board): Returns a new board with one 2x2 block flipped.

    Safeguards:
        - Raises ValueError if box_size is None.
        - Ensures block indices are within bounds (avoids overflow).
        - Clones the board to prevent modifying input.
    """
    def __init__(self, box_size: Optional[Tuple[int,int]] = None, name: Optional[str] = None):
        super().__init__(name, box_size=box_size)
        if box_size is None:
            raise ValueError('None')
        

    def propose(self, board: Board) -> Board:
        b = board.clone()
        N, H, W = b.shape

        # Define bounding box
        if self.box_size:
            Ah, Aw = self.box_size
            h_start = (H - Ah) // 2
            w_start = (W - Aw) // 2
            h_end = h_start + Ah - 1
            w_end = w_start + Aw - 1
        else:
            h_start, h_end = 0, H-1
            w_start, w_end = 0, W-1

        # Random indices inside box
        idx = torch.randint(0, N, (1,)).item()
        i = torch.randint(h_start, h_end, (1,)).item()
        j = torch.randint(w_start, w_end, (1,)).item()

        b._states[idx, i:i+2, j:j+2] ^= 1
        return b