import math
import os
import pickle
import random
from collections import namedtuple, deque

import numpy as np
import torch

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0
steps_per_game = 100

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    global steps_done
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
    self.action_deque = deque(maxlen=2)
    self.action_deque.append("NONE")
    self.action_deque.append("NONE")
    self.exploration_map = None
    steps_done = 0
    # self.policy_net = None
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file)


def get_valid_actions(self, game_state: dict):
    tmp_field = game_state['field']
    x, y = game_state['self'][3]
    preferred_actions = []
    discouraged_actions = []
    if self.exploration_map[(x, y+1)] == 0:
        preferred_actions.append('DOWN')
    if self.exploration_map[(x, y+1)] == 1:
        discouraged_actions.append('DOWN')

    if self.exploration_map[(x+1, y)] == 0:
        preferred_actions.append('RIGHT')
    if self.exploration_map[(x+1, y)] == 1:
        discouraged_actions.append('RIGHT')

    if self.exploration_map[(x, y-1)] == 0:
        preferred_actions.append('UP')
    if self.exploration_map[(x, y-1)] == 1:
        discouraged_actions.append('UP')

    if self.exploration_map[(x-1, y)] == 0:
        preferred_actions.append('LEFT')
    if self.exploration_map[(x-1, y)] == 1:
        discouraged_actions.append('LEFT')
    return preferred_actions, discouraged_actions

def choose_best_possible_action(self, game_state: dict, action):
    np_action = np.argsort(action.numpy())[::-1]
    preferred_actions, discouraged_actions = get_valid_actions(self, game_state)
    for i in np_action:
        if (ACTIONS[i] in preferred_actions) and (ACTIONS[i] != self.action_deque[0]): # discourage repeat behaviour
            self.action_deque.append(ACTIONS[i])
            #print(self.action_deque)
            #print(ACTIONS[i])
            return ACTIONS[i]
    for i in np_action:
        if (ACTIONS[i] in discouraged_actions) and (ACTIONS[i] != self.action_deque[0]): # discourage repeat behaviour
            #print(f"self.action_deque[1]: {self.action_deque[0]}")
            self.action_deque.append(ACTIONS[i])
            #print(self.action_deque)
            #print(ACTIONS[i])
            return ACTIONS[i]
    print("Could not find correct stuff")





def act(self, game_state: dict) -> str:
    global steps_done
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if (steps_done % steps_per_game) == 0:
        self.exploration_map = game_state['field']
    self.exploration_map[game_state['self'][3]] = 1
    # print(f"self.exploration_map: {self.exploration_map}")
    steps_done += 1
    # get_valid_actions(self, game_state)
    if sample > eps_threshold or not self.train:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            features = state_to_features(game_state)
            features_tensor = torch.from_numpy(features).float()
            action = self.policy_net(features_tensor)
            # return ACTIONS[torch.argmax(action)]
            chosen_action = choose_best_possible_action(self, game_state, action)
            return chosen_action
    else:
        preferred_actions, discouraged_actions = get_valid_actions(self, game_state)
        v_actions = preferred_actions + discouraged_actions
        #print(f"rand: {v_actions[random.randrange(len(v_actions))]}")
        #print(f"v_actions: {v_actions}")
        return np.random.choice(v_actions)
        #np.random.choice(ACTIONS, p=self.model)


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
    hybrid_matrix = np.zeros(field_shape + (6,), dtype=np.double)

    # Others
    for _, _, _, (x, y) in game_state["others"]:
        hybrid_matrix[x, y, 0] = 1

    # Bombs
    for (x, y), _ in game_state["bombs"]:
        hybrid_matrix[x, y, 1] = 1

    # Coins
    for (x, y) in game_state["coins"]:
        hybrid_matrix[x, y, 2] = 1

    # Crates
    hybrid_matrix[:, :, 3] = np.where(game_state["field"] == 1, 1, 0)

    # Walls
    hybrid_matrix[:, :, 4] = np.where(game_state["field"] == -1, 1, 0)

    # Position of user
    _, _, _, (x, y) = game_state["self"]
    hybrid_matrix[x, y, 5] = 1

    return hybrid_matrix.reshape(-1)
