import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import torch
from torch import optim
import torch.nn.functional as F

import events as e

# This is only an example!
import settings
from .callbacks import state_to_features, ACTIONS
from .model import DQN
from .replay_memory import ReplayMemory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 4

# Events
# Defined by ICEC 2019
LAST_MAN_STANDING = "LAST_MAN_STANDING"
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
CLOSEST_TO_ENEMY = "CLOSEST_TO_ENEMY"
FARTHER_TO_ENEMY = "FARTHER_TO_ENEMY"
DANGER_ZONE_BOMB = "DANGER_ZONE_BOMB"
SAFE_CELL_BOMB = "SAFE_CELL_BOMB"
ALREADY_VISITED_EVENT = "ALREADY_VISITED_EVENT"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    n_actions = 6

    self.policy_net = DQN(867, n_actions)
    self.target_net = DQN(867, n_actions)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.visited = np.zeros((17, 17))
    self.visited_before = np.zeros((17, 17))

    self.steps_done = 0
    self.rounds = 0

    self.optimizer = optim.RMSprop(self.policy_net.parameters())
    self.memory = ReplayMemory(100000)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Distances to enemies
    if old_game_state:
        _, _, _, pos_old = old_game_state["self"]

    _, _, _, pos_current = new_game_state["self"]

    # Bomb blast range
    # TODO What happens if two bombs are in reach of current position?
    current_bombs = new_game_state["bombs"]
    is_getting_bombed = False
    for (x, y), countdown in current_bombs:
        for i in range(0, settings.BOMB_POWER + 1):
            if new_game_state['field'][x + i, y] == -1:
                break
            # Check current position
            if pos_current == (x + i, y):
                is_getting_bombed = True
        for i in range(0, settings.BOMB_POWER + 1):
            if new_game_state['field'][x - i, y] == -1:
                break
            # Check current position
            if pos_current == (x - i, y):
                is_getting_bombed = True
        for i in range(0, settings.BOMB_POWER + 1):
            if new_game_state['field'][x, y + i] == -1:
                break
            # Check current position
            if pos_current == (x, y + i):
                is_getting_bombed = True
        for i in range(0, settings.BOMB_POWER + 1):
            if new_game_state['field'][x, y - i] == -1:
                break
            # Check current position
            if pos_current == (x, y - i):
                is_getting_bombed = True

    if is_getting_bombed:
        events.append(DANGER_ZONE_BOMB)
        self.logger.debug(f'Add game event {DANGER_ZONE_BOMB} in step {new_game_state["step"]}')
    else:
        events.append(SAFE_CELL_BOMB)
        self.logger.debug(f'Add game event {SAFE_CELL_BOMB} in step {new_game_state["step"]}')

    if self.visited_before[pos_current[0]][pos_current[1]] == 1:
        events.append(ALREADY_VISITED_EVENT)
        self.logger.debug(f'Add game event {ALREADY_VISITED_EVENT} in step {new_game_state["step"]}')

    self.visited_before = self.visited
    self.visited[pos_current[0]][pos_current[1]] = 1

    if old_game_state is not None:
        self.memory.push(state_to_features(old_game_state), [ACTIONS.index(self_action)],
                         state_to_features(new_game_state),
                         reward_from_events(self, events))

        optimize_model(self)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), [ACTIONS.index(last_action)], None,
                   reward_from_events(self, events)))

    self.visited = np.zeros((17, 17))
    self.visited_before = np.zeros((17, 17))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net, file)

    # Update the target network, copying all weights and biases in DQN
    if self.rounds % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = 0

    self.rounds += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.CRATE_DESTROYED: 1,
        e.KILLED_SELF: -25,
        e.INVALID_ACTION: -1,
        e.WAITED: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # Penalty per iteration
    reward_sum -= 0.1

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def optimize_model(self):
    if len(self.memory) < BATCH_SIZE + 1:
        return

    transitions = self.memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)

    state_batch = torch.tensor(batch.state).float()
    next_state_batch = torch.tensor(batch.next_state).float()
    reward_batch = torch.tensor(batch.reward).float()

    non_final_next_states = torch.cat([s for s in next_state_batch
                                       if s is not None])

    action_batch = torch.tensor(np.asarray(batch.action, dtype=np.int64))
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states.reshape(-1, 578)).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()
