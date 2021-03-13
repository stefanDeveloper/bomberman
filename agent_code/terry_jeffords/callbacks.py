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
# how to import nn
# get linear stuff


class torchmodel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(torchmodel, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(dim_in, 512),
            #nn.ReLU(),
            nn.Linear(512, 512),
            #nn.ReLU(),
            nn.Linear(512, dim_out),
            #nn.ReLU()
        )

    def forward(self, x):
        logits = self.model_sequence(x)
        return logits


def find_forbidden_moves(field, pos):
    # pos switches x and y for visibility, but this is confusing for the actual array
    y, x = pos[0], pos[1]
    up = (x-1, y)
    down = (x+1, y)
    right = (x, y+1)
    left = (x, y-1)
    allowed_move_i = np.zeros(len(ACTIONS))
    # print(field)
    # print(pos)
    # finds moves, which are forbidden and allows to move to empty space
    if field[up] == 0:
        allowed_move_i[0] = 1
    if field[right] == 0:
        allowed_move_i[1] = 1
    if field[down] == 0:
        allowed_move_i[2] = 1
    if field[left] == 0:
        allowed_move_i[3] = 1
    allowed_move_i[4] = allowed_move_i[5] = 1
    return np.array(allowed_move_i, dtype=bool)

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
        self.testmodel = torchmodel(1445, 6)
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
    # print(state_to_features(game_state).shape) currently 17*17, 6 channels
    features = state_to_features_icaart(game_state)
    model_input = torch.from_numpy(features).float()
    # print(torch.sum(model_input))
    model_output = self.testmodel(model_input)

    # Restrict possible moves via empty spaces in field
    allowed_moves = find_forbidden_moves(game_state["field"], game_state["self"][3])
    # index of most likely allowed move
    outindex = torch.argmax(model_output[allowed_moves])
    # inefficient helper function, since mask does not work for python list
    allowed_actions = []
    for i in range(len(ACTIONS)):
        if allowed_moves[i]:
            allowed_actions.append(ACTIONS[i])
    print(allowed_actions)
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    # just let a peaceful agent run for getting game states
    # return np.random.choice(ACTIONS[:-1], p=[.3, .2, .2, .2, .1])
    return allowed_actions[outindex]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*
    Converts the game state to the input of your model, i.e.
    a feature vector.
    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to
    see what it contains.
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # print(game_state["field"].shape)

    field_channel = np.array(game_state["field"])
    field_shape = field_channel.shape
    # create new "maps" for self, others, bombs, coins and explosion_map
    # self (only has one entry)
    self_channel = np.zeros(field_shape)
    self_channel[game_state["self"][3]] = 1

    # others
    others_channel = np.zeros(field_shape)
    for i in game_state["others"]:
        others_channel[i[3]] = -1
    # bombs
    bombs_channel = np.zeros(field_shape)
    for i in game_state["bombs"]:
        bombs_channel[i[0]] = i[1]

    # coins
    coins_channel = np.zeros(field_shape)
    for i in game_state["coins"]:
        coins_channel[i] = 1

    # explosion_map
    explosion_channel = np.array(game_state["explosion_map"])
    # For example, you could construct several channels of equal shape, ...
    channels = [field_channel, self_channel, others_channel, bombs_channel, coins_channel, explosion_channel]
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


