import numpy as np


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
    field_shape = game_state["field"].shape

    # Create Hybrid Matrix with field shape x vector of size 5 to encode field state
    hybrid_matrix = np.zeros((3,) + field_shape, dtype=np.double)

    # Coins
    for (x, y) in game_state["coins"]:
        hybrid_matrix[0, x, y] = 1

    # Walls
    hybrid_matrix[2, :, :] = np.where(game_state["field"] == -1, 1, 0)

    # Position of user
    _, _, _, (x, y) = game_state["self"]
    hybrid_matrix[1, x, y] = 1

    return hybrid_matrix  # return the map (batch_size, channels, height, width)
