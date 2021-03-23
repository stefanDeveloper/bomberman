import numpy as np
from scipy.ndimage import shift

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    # One-Hot encoding
    fx, fy = game_state["field"].shape
    # Create Hybrid Matrix with field shape x vector of size 5 to encode field state
    hybrid_matrix = np.zeros((5, fx -2, fy - 2), dtype=np.double)

    # Others
    for _, _, _, (x, y) in game_state["others"]:
        hybrid_matrix[0, x - 1, y - 1] = 1
    hybrid_matrix[0] = hybrid_matrix[0].T

    # Bombs
    for (x, y), countdown in game_state["bombs"]:
       hybrid_matrix[1, x - 1, y - 1] = countdown
    hybrid_matrix[1] = hybrid_matrix[1].T

    # Coins
    for (x, y) in game_state["coins"]:
        hybrid_matrix[2, x - 1, y - 1] = 1
    hybrid_matrix[2] = hybrid_matrix[2].T

    # Crates
    hybrid_matrix[3, :, :] = np.where(game_state["field"][1:-1, 1:-1] == 1, 1, 0)

    # Walls
    hybrid_matrix[4, :, :] = np.where(game_state["field"][1:-1, 1:-1] == -1, 1, 0)

    # Position of user
    _, _, _, (x, y) = game_state["self"]

    dx = 8 - x
    dy = 8 - y

    for i in range(len(hybrid_matrix)):
        hybrid_matrix[i] = shift(input=hybrid_matrix[i], shift=(dy, dx), order=0)

    hybrid_matrix = hybrid_matrix[:, 2:-2, 2:-2]
    
    return hybrid_matrix  # return the map (batch_size, channels, height, width)
