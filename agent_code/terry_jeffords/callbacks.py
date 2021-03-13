import os
import pickle
import random
from collections import deque

import numpy as np

import torch

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
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        #self.test_model = DQN(1734, 6)
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
    # todo Exploration vs exploitation
    self.logger.info('Picking action according to rule set')

    features = state_to_features(game_state)

    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    # TODO Call get action
    #return np.random.choice(ACTIONS, p=self.model)
    return allowed_actions[outindex]



import os
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import settings

import numpy as np


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
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    # just let a peaceful agent run for getting game states
    return np.random.choice(ACTIONS[:-1], p=[.3, .2, .2, .2, .1])

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
    hybrid_matrix = np.zeros((field_shape) + (5,))

    # others
    for i in game_state["others"]:
        hybrid_matrix[i[3], 0] = -1

    # bombs
    for i in game_state["bombs"]:
        hybrid_matrix[i[0], 1] = i[1]

    # coins
    for i in game_state["coins"]:
        hybrid_matrix[i, 2] = 1

    # crates
    hybrid_matrix[:, :, 3] = np.where(game_state["field"] == 1, 1, 0)

    # walls
    hybrid_matrix[:, :, 4] = np.where(game_state["field"] == -1, 1, 0)

    # explosion_map
    # explosion_channel = np.array(game_state["explosion_map"])
    # For example, you could construct several channels of equal shape, ...
    # channels = [field_channel, self_channel, others_channel, bombs_channel, coins_channel, explosion_channel]
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # and return them as a vector
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
                              field_channel)
    # For example, you could construct several channels of equal shape, ...
    channels = [field_channel, self_channel, others_channel, danger_channel, coins_channel]
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    final_vector = np.stack(channels).reshape(-1)
    print(final_vector.shape)
    # and return them as a vector
    return final_vector


