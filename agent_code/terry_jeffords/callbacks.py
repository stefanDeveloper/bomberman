import os
import pickle
import random

import numpy as np
import torch

import settings
from agent_code.terry_jeffords.model import DQN

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
    if self.train or not os.path.isfile("terry-jeffords-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DQN(1447, 6)
    else:
        self.logger.info("Loading model from saved state.")
        with open("terry-jeffords-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # TODO Exploration vs exploitation
    self.logger.info('Picking action according to rule set')

    if self.train and random.random() < self.exploration_rate:
        self.logger.debug("Choosing action purely at random.")
        self.logger.debug(f'Exploration rate: {self.exploration_rate}')
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    features = state_to_features_hybrid(game_state)
    features_tensor = torch.from_numpy(features).float()
    predicted_reward = self.model(features_tensor)
    action = torch.argmax(predicted_reward)

    self.logger.info(f'Selected action: {action}')

    return ACTIONS[action]


def state_to_features_hybrid(game_state: dict) -> np.array:
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
    hybrid_matrix = np.zeros(field_shape + (5,), dtype=np.double)

    # others
    for i in game_state["others"]:
        hybrid_matrix[i[3], 0] = 1

    # bombs
    for i in game_state["bombs"]:
        hybrid_matrix[i[0], 1] = 1

    # coins
    for i in game_state["coins"]:
        hybrid_matrix[i, 2] = 1

    # crates
    hybrid_matrix[:, :, 3] = np.where(game_state["field"] == 1, 1, 0)

    # walls
    hybrid_matrix[:, :, 4] = np.where(game_state["field"] == -1, 1, 0)

    final_vector = np.append(hybrid_matrix.reshape(-1), (game_state["self"][3]))

    return final_vector


def state_to_features_icaart(game_state: dict) -> np.array:
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

    # Create Field Matrix with field shape
    field_channel = np.zeros(field_shape)

    # Empty cells = 1
    field_channel = np.where(game_state["field"] == 0, 1, field_channel)

    # Wall/Bomb cells = -1
    field_channel = np.where(game_state["field"] == -1, -1, field_channel)
    for i in game_state["bombs"]:
        field_channel[i[0]] = -1

    # others
    others_channel = np.zeros(field_shape)
    for i in game_state["others"]:
        others_channel[i[3]] = 1

    # self (only has one entry)
    self_channel = np.zeros(field_shape)
    self_channel[game_state["self"][3]] = 1

    # coins
    coins_channel = np.zeros(field_shape)
    for i in game_state["coins"]:
        coins_channel[i] = 1

    # danger_channel
    danger_channel = np.zeros(field_shape)
    danger_channel = np.where(game_state["explosion_map"] > 0, settings.BOMB_TIMER / game_state["explosion_map"],
                              danger_channel)

    # Sum up
    channels = [field_channel, self_channel, others_channel, danger_channel, coins_channel]

    final_vector = np.stack(channels).reshape(-1)

    print(f'Feature has shape of {final_vector.shape}')

    return final_vector
