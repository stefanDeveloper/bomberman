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
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    if self.train or not os.path.isfile("terry-jeffords-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        self.test_model = DQN(1734, 6)
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
    allowed_move_i = np.zeros(len(ACTIONS))

    features = state_to_features(game_state)
    if game_state is not None:
        # Gather information about the game state
        # used to check wall, crate, free tile
        arena = game_state['field']
        # position of my agent and other agents
        _, score, bombs_left, (x, y) = game_state['self']
        # self.s = 15*(int(x)-1)+(int(y)-1)  #current state for q learning
        others = [(x, y) for (n, s, b, (x, y)) in game_state['others']]
        # bomb
        bombs = game_state['bombs']
        bomb_xys = [(x, y) for ((x, y), t) in bombs]
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        # coin
        coins = game_state['coins']

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        # find valid actions
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []

        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] <= 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles:
            valid_actions.append('LEFT')
            allowed_move_i[3] = 1
        if (x + 1, y) in valid_tiles:
            valid_actions.append('RIGHT')
            allowed_move_i[1] = 1
        if (x, y - 1) in valid_tiles:
            valid_actions.append('UP')
            allowed_move_i[0] = 1
        if (x, y + 1) in valid_tiles:
            valid_actions.append('DOWN')
            allowed_move_i[2] = 1
        if (x, y) in valid_tiles:
            valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
        self.logger.debug(f'Valid actions: {valid_actions}')

    allowed_moves = np.array(allowed_move_i, dtype=bool)

    features = state_to_features(game_state)
    model_input = torch.from_numpy(features).float()

    model_output = self.test_model(model_input)
    # index of most likely allowed move
    outindex = torch.argmax(model_output[allowed_moves])

    allowed_actions = []
    for i in range(len(ACTIONS)):
        if allowed_moves[i]:
            allowed_actions.append(ACTIONS[i])

    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    return allowed_actions[outindex]


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
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
