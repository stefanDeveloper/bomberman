import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    learning_rate = 0.003

    def __init__(self, dim_in, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Conv2d(3, 1, 3),  # inchannels, out_channels, kernel size
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

    def forward(self, x):
        logits = self.model_sequence(x.view(-1, 3, 17, 17))
        return logits
