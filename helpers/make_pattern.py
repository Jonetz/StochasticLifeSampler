import numpy as np
import matplotlib.pyplot as plt

def create_custom_board(H=15, W=15, scale=20):
    """
    Launch an interactive grid to select cells.
    Left click: toggle cell alive/dead.
    Close window to finish.
    Returns:
        np.ndarray of shape (H, W) with 0/1 entries.
    """
    board = np.zeros((H, W), dtype=np.uint8)
    
    fig, ax = plt.subplots()
    im = ax.imshow(board, cmap='gray_r', vmin=0, vmax=1, interpolation='none')
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(length=0)
    
    def onclick(event):
        if event.inaxes == ax:
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)
            if 0 <= x < W and 0 <= y < H:
                board[y, x] = 1 - board[y, x]  # toggle
                im.set_data(board)
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click cells to toggle. Close window when done.")
    plt.show()
    
    return board

def print_with_commas(arr: np.ndarray):
    """
    Print a NumPy array with commas between elements and full expansion.
    """
    s = np.array2string(
        arr,
        separator=', ',
        threshold=np.inf,  # avoid truncation with ...
        max_line_width=np.inf
    )
    print(s)

custom_board = create_custom_board(H=50, W=50)
np.set_printoptions(threshold=np.inf)

print_with_commas(custom_board)
