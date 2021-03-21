from collections import deque
from typing import Tuple, List

import numpy as np


def state_to_features(game_state: dict, action: str) -> np.array:
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

    channels = []

    _, _, _, position = game_state["self"]
    new_position = position
    if action == 'LEFT' and game_state["field"][position[0] - 1, position[1]] != -1:
        new_position = position[0] - 1, position[1]
    elif action == 'RIGHT' and game_state["field"][position[0] + 1, position[1]] != -1:
        new_position = position[0] + 1, position[1]
    elif action == 'UP' and game_state["field"][position[0], position[1] - 1] != -1:
        new_position = position[0], position[1] - 1
    elif action == 'DOWN' and game_state["field"][position[0], position[1] + 1] != -1:
        new_position = position[0], position[1] + 1

    # Calculate distance to coins
    coin_bfs = CoinBFS(game_state["field"], game_state['coins'])
    distance, coin_position = coin_bfs.get_distance(new_position)
    channels.append(distance)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


class CoinBFS:
    def __init__(self, field: np.array, coins: List[Tuple[int, int]]):
        self.field = field

        self.row = [-1, 0, 0, 1]
        self.col = [0, -1, 1, 0]

        self.N = field.shape[1]
        self.M = field.shape[0]

        self.visited = [[False for _ in range(self.N)] for _ in range(self.M)]
        self.queue = deque()

        for coin in coins:
            x, y = coin
            self.field[x, y] = 2

    def is_valid(self, row: int, col: int) -> bool:
        return (row >= 0) and (row < self.M) and (col >= 0) and (col < self.N) and self.field[row][col] != -1 and not \
            self.visited[row][col]

    def get_distance(self, position: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
        x, y = position

        # Mark the source cell as visited
        self.visited[x][y] = True

        # Enqueue the source node
        self.queue.append((x, y, 0))

        # Store min distance and coordinates of nearest coin
        min_dist = float("inf")
        min_coin = (0, 0)

        # Loop until queue is empty
        while self.queue:
            # Dequeue first element
            (x, y, dist) = self.queue.popleft()

            # Check if current field is a coin
            if self.field[x, y] == 2:
                min_dist = dist
                min_coin = (x, y)
                break

            # Iterate over all possible movements (up, down, left, right)
            for k in range(4):
                # Check if field is valid field
                if self.is_valid(x + self.row[k], y + self.col[k]):
                    # Mark field as visited
                    self.visited[x + self.row[k]][y + self.col[k]] = True
                    # Enqueue field
                    self.queue.append((x + self.row[k], y + self.col[k], dist + 1))

        # Return min distance and coordinates of the coin
        return min_dist, min_coin
