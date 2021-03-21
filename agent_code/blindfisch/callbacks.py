import os
import pickle
import random
import torch as T
import torch.nn.functional as F
from .model import DQN
import numpy as np
from .StateToFeat import state_to_features
from .callbacks_rule import act_rule


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
    self.global_counter = 0
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model")
        self.model = DQN(4, 6)
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
    self.global_counter += 1
    random_prob = 1.0
    if self.global_counter > 400 * 100:
        random_prob = 0.3
    if self.global_counter > 400 * 200:
        random_prob = 0.1
    if self.global_counter > 400 * 400:
        random_prob = 0.0

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # return act_rule(game_state)
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1, .0])
    self.logger.debug("Querying model for action.")
    #if not self.train:
        #return ACTIONS[np.argmax(self.model.forward(game_state), dim=1)[0].detach().numpy()]
    #print(f"output: {F.softmax(self.model.forward(game_state), dim=1)[0].detach().numpy()}" )
    #print(np.random.choice(ACTIONS, p=F.softmax(self.model.forward(game_state), dim=1)[0].detach().numpy()))
    # print(self.model.forward(game_state))
    # print(F.softmax(self.model.forward(game_state), dim=0).detach().numpy())
    res = np.random.choice(ACTIONS, p=F.softmax(self.model.forward(game_state), dim=0).detach().numpy())
    # print(res)
    return res
    # return act_rule(game_state) # so this works nicely

# Here was state_to_features
