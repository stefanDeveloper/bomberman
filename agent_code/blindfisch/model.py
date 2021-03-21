import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
# from .callbacks import state_to_features # probably do this instead or change state_to_features to here
from .StateToFeat import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class DQN(nn.Module):
    gamma = 0.9  # 0.9
    learning_rate = 0.0003

    def __init__(self, dim_in, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(dim_in, dim_out)
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = T.tensor(state_to_features(x)).float()
        return self.model_sequence(x.to(self.device)).to('cpu')

    def train_step(self, old_state, action, new_state, reward):
        if action is not None:
            action_mask = T.zeros(len(ACTIONS), dtype=T.int64)
            action_mask[ACTIONS.index(action)] = 1

            state_action_value = T.masked_select(self.forward(old_state), action_mask.bool())
            next_state_action_value = self.forward(new_state).max().unsqueeze(0)
            expected_state_action_value = (next_state_action_value * self.gamma) + reward

            loss = self.loss(state_action_value.to(self.device), expected_state_action_value.to(self.device))
            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_step_bad(self, old_state, action, new_state, reward):
        if action is not None:
            reward_tensor = T.zeros(len(ACTIONS), dtype=T.int64)
            reward_tensor[ACTIONS.index(action)] = reward

            state_action_value = self.forward(old_state)
            next_state_action_value = self.forward(new_state)
            expected_state_action_value = (next_state_action_value * self.gamma) + reward

            loss = self.loss(state_action_value.to(self.device), expected_state_action_value.to(self.device))
            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_step_very_bad(self, old_state, action, new_state, reward):
        if action is not None:
            action_mask = T.zeros(len(ACTIONS), dtype=T.int64)
            choice_range = np.arange(len(ACTIONS), dtype=np.int64)
            choice_mask = T.zeros(len(ACTIONS), dtype=T.int64)
            action_mask[ACTIONS.index(action)] = 1
            next_state_val = self.forward(new_state)
            chosen_index = np.random.choice(choice_range, p=F.softmax(next_state_val, dim=1)[0].detach().numpy())
            choice_mask[chosen_index] = 1
            # np.random.choice(ACTIONS, p=F.softmax(self.model.forward(game_state), dim=1)[0].detach().numpy())
            # p=F.softmax(self.model.forward(game_state), dim=1)[0].detach().numpy()

            state_action_value = T.masked_select(self.forward(old_state), action_mask.bool())
            next_state_action_value = T.masked_select(next_state_val, choice_mask.bool())
            # next_state_action_value = self.forward(new_state).max().unsqueeze(0)
            expected_state_action_value = (next_state_action_value * self.gamma) + reward

            loss = self.loss(state_action_value.to(self.device), expected_state_action_value.to(self.device))
            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
