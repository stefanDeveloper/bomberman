import pickle
from collections import namedtuple, deque
from typing import List

import numpy as np

import events as e
import settings
from agent_code.terry_jeffords.callbacks import state_to_features_hybrid

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EXPLORATION_RATE = 0.1

# Events
# Defined by ICEC 2019
LAST_MAN_STANDING = "LAST_MAN_STANDING"
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
CLOSEST_TO_ENEMY = "CLOSEST_TO_ENEMY"
FARTHER_TO_ENEMY = "FARTHER_TO_ENEMY"
DANGER_ZONE_BOMB = "DANGER_ZONE_BOMB"
SAFE_CELL_BOMB = "SAFE_CELL_BOMB"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.exploration_rate = EXPLORATION_RATE


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
        pos_enemy_old = np.array([pos for _, _, _, pos in old_game_state['others']])
        enemy_distance_old = np.sum(np.abs(np.subtract(pos_enemy_old, pos_old)), axis=1).min()

    _, _, _, pos_current = new_game_state["self"]
    pos_enemy_current = np.array([pos for _, _, _, pos in new_game_state['others']])
    enemy_distance_current = np.sum(np.abs(np.subtract(pos_enemy_current, pos_current)), axis=1).min()

    # First round, set min distance to enemies
    if new_game_state["round"] == 1:
        self.min_enemy_distance = enemy_distance_current

    # Idea: Add your own events to hand out rewards
    if len(new_game_state["others"]) == 0:
        events.append(LAST_MAN_STANDING)
        self.logger.debug(f'Add game event {LAST_MAN_STANDING} in step {new_game_state["step"]}')

    if old_game_state and enemy_distance_current < enemy_distance_old:
        events.append(CLOSER_TO_ENEMY)
        self.logger.debug(f'Add game event {CLOSER_TO_ENEMY} in step {new_game_state["step"]}')

    if enemy_distance_current < self.min_enemy_distance:
        events.append(CLOSEST_TO_ENEMY)
        self.min_enemy_distance = enemy_distance_current
        self.logger.debug(f'Add game event {CLOSEST_TO_ENEMY} in step {new_game_state["step"]}')

    if old_game_state and enemy_distance_current > enemy_distance_old:
        events.append(FARTHER_TO_ENEMY)
        self.logger.debug(f'Add game event {FARTHER_TO_ENEMY} in step {new_game_state["step"]}')

    # Bomb blast range
    # TODO What happens if two bombs are in reach of current position?
    current_bombs = new_game_state["bombs"]
    is_getting_bombed = False
    for (x, y), countdown in current_bombs:
        for i in range(1, settings.BOMB_POWER + 1):
            if new_game_state['field'][x + i, y] == -1:
                break
            if pos_current == (x + i, y):
                is_getting_bombed = True
        for i in range(1, settings.BOMB_POWER + 1):
            if new_game_state['field'][x - i, y] == -1:
                break
            if pos_current == (x - i, y):
                is_getting_bombed = True
        for i in range(1, settings.BOMB_POWER + 1):
            if new_game_state['field'][x, y + i] == -1:
                break
            if pos_current == (x, y + i):
                is_getting_bombed = True
        for i in range(1, settings.BOMB_POWER + 1):
            if new_game_state['field'][x, y - i] == -1:
                break
            if pos_current == (x, y - i):
                is_getting_bombed = True

    if is_getting_bombed:
        events.append(DANGER_ZONE_BOMB)
        self.logger.debug(f'Add game event {DANGER_ZONE_BOMB} in step {new_game_state["step"]}')
    else:
        events.append(SAFE_CELL_BOMB)
        self.logger.debug(f'Add game event {SAFE_CELL_BOMB} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features_hybrid(old_game_state), self_action, state_to_features_hybrid(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.

    Args:
        events:
        last_action:
        last_game_state:
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features_hybrid(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -0.5,
        e.CRATE_DESTROYED: 0.1,
        LAST_MAN_STANDING: 1,
        CLOSER_TO_ENEMY: 0.002,
        CLOSEST_TO_ENEMY: 0.1,
        FARTHER_TO_ENEMY: -0.002,
        DANGER_ZONE_BOMB: -0.000666,
        SAFE_CELL_BOMB: 0.002,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    # Penalty per iteration
    reward_sum -= 0.01

    return reward_sum
