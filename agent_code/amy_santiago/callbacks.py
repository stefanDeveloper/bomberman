import math
import os
import pickle
import random

import numpy as np
import torch

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.steps_done = 0
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
    self.steps_done += 1
    if self.train:
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                features = state_to_features(game_state)
                features_tensor = torch.from_numpy(features).float()
                action = self.policy_net(features_tensor)
                return ACTIONS[torch.argmax(action)]
        else:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        features = state_to_features(game_state)
        features_tensor = torch.from_numpy(features).float()
        action = self.policy_net(features_tensor)
        return ACTIONS[torch.argmax(action)]


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

    field_shape = game_state["field"].shape

    # Create Hybrid Matrix with field shape x vector of size 5 to encode field state
    hybrid_matrix = np.zeros(field_shape + (3,), dtype=np.double)

    # Others
    #for _, _, _, (x, y) in game_state["others"]:
    #    hybrid_matrix[x, y, 0] = 1

    # Bombs
    #for (x, y), _ in game_state["bombs"]:
    #    hybrid_matrix[x, y, 1] = 1

    # Coins
    for (x, y) in game_state["coins"]:
        hybrid_matrix[x, y, 0] = 1

    # Crates
    #hybrid_matrix[:, :, 3] = np.where(game_state["field"] == 1, 1, 0)

    # Walls
    hybrid_matrix[:, :, 1] = np.where(game_state["field"] == -1, 1, 0)

    # Position of user
    _, _, _, (x, y) = game_state["self"]
    hybrid_matrix[x, y, 2] = 1

    return hybrid_matrix.reshape(-1)
