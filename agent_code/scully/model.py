import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
# from .callbacks import state_to_features # probably do this instead or change state_to_features to here
from .StateToFeat import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class DQN(nn.Module):
    gamma = 0.9
    learning_rate = 0.0003

    def __init__(self, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Conv2d(2, 1, 3),  # inchannels, out_channels, kernel size
            nn.Flatten(start_dim=1),
            nn.Linear(15*15, 128),  # def 2048, 512, 15*15 is image size after conv
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, dim_out),
            nn.Softmax(dim=1)
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        #self.to(self.device)

    def forward(self, x):
        x = T.tensor(state_to_features(x)).float().view(-1, 2, 17, 17) # batch_size, channels, img_dims
        # print("I got here")
        # print(self.model_sequence(x))
        return self.model_sequence(x)

    def train_step(self, old_state, action, new_state, reward):
        if action is not None:
            action_mask = T.zeros(len(ACTIONS), dtype=T.int64)
            action_mask[ACTIONS.index(action)] = 1

            state_action_value = T.masked_select(self.forward(old_state), action_mask.bool())
            next_state_action_value = self.forward(new_state).max().unsqueeze(0)
            expected_state_action_value = (next_state_action_value * self.gamma) + reward

            loss = self.loss(state_action_value, expected_state_action_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
