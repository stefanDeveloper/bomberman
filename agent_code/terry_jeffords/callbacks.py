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
    # TODO Have a nice feature representation
    # We want to represent the state as vector.
    # For each cell on the field we define a vector with 5 entries, each either 0 or 1
    # [0, 0, 0, 0, 0] --> free
    # [1, 0, 0, 0, 0] --> stone
    # [0, 1, 0, 0, 0] --> crate
    # [0, 0, 1, 0, 0] --> coin
    # [0, 0, 0, 1, 0] --> bomb
    # [0, 0, 0, 0, 1] --> fire
    # in principle with this encoding multiple cases could happen at the same time
    # e.g. [0, 0, 0, 1, 1] --> bomb and fire
    # but in our implementation of the game this is not relevant
    # because they are a combination of one-hot and binary map
    # they are called hybrid vectors

    # initialize empty field
    # note: in the game we have a field of 17x17, but the borders are always
    # stone so we reduce the dimension to 15x15
    hybrid_vectors = np.zeros((15, 15, 5), dtype=int)

    # check where there are stones on the field
    # just use the field without the borders (1:-1)
    # set the first entry in the vector to 1
    hybrid_vectors[np.where(game_state['field'][1:-1, 1:-1] == -1), 0] = 1

    # check where there are crates
    # set the second entry in the vector to 1
    hybrid_vectors[np.where(game_state['field'][1:-1, 1:-1] == 1), 1] = 1

    # check where free coins are
    # set the third entry in the vector to 1
    # user np.moveaxis to transform list of tuples in numpy array
    # https://stackoverflow.com/questions/42537956/slice-numpy-array-using-list-of-coordinates
    # -1 in coordintaes because we left out the border
    coin_coords = np.moveaxis(np.array(game_state['coins']), -1, 0)
    hybrid_vectors[coin_coords[0] - 1, coin_coords[1] - 1, 2] = 1

    # check where bombs are
    # set the fourth entry in the vector to 1
    # discard the time since this can be learned by the model because we
    # use a LSTM network
    bomb_coords = np.array([[bomb[0][0], bomb[0][1], bomb[1]] for bomb in game_state['bombs']]).T
    hybrid_vectors[bomb_coords[0] - 1, bomb_coords[1] - 1, 3] = bomb_coords[2]

    # check where fire is
    # set the fifth entry in the vector to 1
    hybrid_vectors[:, :, 4] = game_state['explosion_map'][1:-1, 1:-1]

    # flatten 3D array to 1D vector
    hyb_vec = hybrid_vectors.flatten()

    # add enemy coords and their bomb boolean as additional entries at the end
    # non-existing enemies have -1 at each position as default
    for i in range(3):
        if len(game_state['others']) > i:
            enemy = game_state['others'][i]
            hyb_vec = np.append(hyb_vec, [enemy[3][0], enemy[3][1], int(enemy[2])])
        else:
            hyb_vec = np.append(hyb_vec, [-1, -1, -1])

    # add own position and availability of bomb as 3 additional entries at the end
    hyb_vec = np.append(hyb_vec, [game_state['self'][3][0], game_state['self'][3][1], int(game_state['self'][2])])

    return hyb_vec  # len(hyb_vec) = (15 x 15 x 5) + (4 x 3) = 1137
