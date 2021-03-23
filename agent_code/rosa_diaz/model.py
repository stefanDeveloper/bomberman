import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    gamma = 1.0
    learning_rate = 0.0003

    def __init__(self, dim_in, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(dim_in, 6),  # def 2048, 512
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    def forward(self, x):
        return self.model_sequence(x)
