import numpy as np
from scipy import signal

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
    # TODO: Coin and danger "heatmap" with convolution and only give immediate surroundings
    if game_state is None:
        return None
    _, _, _, (x_me, y_me) = game_state["self"]
    # One-Hot encoding
    field_shape = game_state["field"].shape
    walls_n_stuff = game_state["field"]

    # Coins
    coin_field = np.zeros(field_shape, dtype=np.double)
    for (x, y) in game_state["coins"]:
        coin_field[x, y] = 10  # this is a coin matrix
    coin_kernel = np.array([[0., 0., 0.125, 0., 0.], [0., 0.125, 0.25, 0.125, 0.], [0.125, 0.25, 1., 0.25, 0.125], [0., 0.125, 0.25, 0.125, 0.], [0., 0., 0.125, 0., 0.]])
    # print("Coin field before convolution")
    # print(coin_field)
    for i in range(5):  # simple convolution
        coin_field = signal.convolve2d(coin_field, coin_kernel, mode='same')
    # print("Coin field after convolution")
    # print(coin_field)
    # these are basically the input to the network
    up_coins = coin_field[x_me, y_me-1]
    down_coins = coin_field[x_me, y_me+1]
    left_coins = coin_field[x_me-1, y_me]
    right_coins = coin_field[x_me+1, y_me]

    if walls_n_stuff[x_me, y_me-1] == -1:
        up_coins = -1
    if walls_n_stuff[x_me, y_me+1] == -1:
        down_coins = -1
    if walls_n_stuff[x_me-1, y_me] == -1:
        left_coins = -1
    if walls_n_stuff[x_me+1, y_me] == -1:
        right_coins = -1

    coin_surroundings = np.array([up_coins, down_coins, left_coins, right_coins])
    # print("coin surroundings")
    # print(coin_surroundings)
    # Create Hybrid Matrix with field shape x vector of size 5 to encode field state
    # hybrid_matrix = np.zeros((2,) + field_shape, dtype=np.double)

    # Others
    # for _, _, _, (x, y) in game_state["others"]:
    #    hybrid_matrix[x, y, 0] = 1

    # Bombs
    # for (x, y), _ in game_state["bombs"]:
    #    hybrid_matrix[x, y, 1] = 1

    # movement matrix
    # Position of user

    # hybrid_matrix[1, x_me, y_me] = 100 # now no longer needed

    # Crates
    # hybrid_matrix[:, :, 3] = np.where(game_state["field"] == 1, 1, 0)

    # Walls
    # hybrid_matrix[2, :, :] = np.where(game_state["field"] == -1, -5, 0)

    # print(hybrid_matrix[:, :, 1])
    # return hybrid_matrix.reshape(-1)
    return coin_surroundings  # return the map (batch_size, channels, height, width)
