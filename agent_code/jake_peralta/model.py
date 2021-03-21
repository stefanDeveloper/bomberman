import math

import numpy as np
from .features import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
TRAINING_RATE = 0.001
GAMMA = 0.8
BATCH_SIZE = 100


def get_action(model: dict, state: dict) -> str:
    # Get Q value for actions
    q_values = []
    for action in ACTIONS:
        feature = state_to_features(state, action)
        q_values.append(feature * model["min_dist"])

    max_q_value = int(np.argmax(q_values))
    return ACTIONS[max_q_value]


def train_step(self, old_state, action, new_state, reward):
    q_values_new = []
    for action in ACTIONS:
        new_feature = state_to_features(new_state, action)
        q_values_new.append(new_feature * self.model["min_dist"])

    max_q_value = np.max(q_values_new)

    old_feature = state_to_features(old_state, action)
    q_values_old = old_feature * self.model["min_dist"]

    temporal_difference = reward + GAMMA * max_q_value - q_values_old
    features = state_to_features(old_state, action)

    self.model["min_dist"] += TRAINING_RATE * temporal_difference * features
