import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
# from .callbacks import state_to_features # probably do this instead or change state_to_features to here
from .StateToFeat import state_to_features
from .callbacks_rule import act_rule

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class DQN(nn.Module):
    gamma = 0.9  # 0.9
    learning_rate = 0.00003

    def __init__(self, dim_in, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            # nn.Conv2d(6, 6, 3),  # inchannels, out_channels, kernel size
            # nn.BatchNorm2d(6),
            # nn.Conv2d(6, 6, 3),
            # nn.BatchNorm2d(6),
            nn.Flatten(start_dim=1),
            nn.Linear(12*12*5, 128),  # def 2048, 512, 15*15 is image size after conv
            nn.ReLU(),
            nn.Linear(128, 6),
            # nn.Softmax(dim=1)
        )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = T.tensor(state_to_features(x)).float().view(-1, 5, 12, 12)
        return self.model_sequence(x.to(self.device)).to('cpu')

    def train_step(self, old_state, action, new_state, reward):
        if action is not None and act_rule(old_state) is not None:
            # action_mask = T.zeros(len(ACTIONS), dtype=T.int64)
            # action_mask[ACTIONS.index(action)] = 1

            state_action_value = self.forward(old_state)# .unsqueeze(0)

            # next_state_action_value = self.forward(new_state).max().unsqueeze(0)
            # expected_state_action_value = (next_state_action_value * self.gamma) + reward
            #print(act_rule(old_state))
            target = T.tensor(ACTIONS.index(act_rule(old_state)), dtype=T.long).unsqueeze(0)
            # print(f"target: {target}")
            # print(f"state_action_value: {state_action_value}")
            loss = self.loss(state_action_value, target).to(self.device)
            # loss = self.loss(state_action_value.to(self.device), expected_state_action_value.to(self.device))
            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
