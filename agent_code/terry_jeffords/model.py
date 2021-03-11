import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(dim_in, 512),
            #nn.ReLU(),
            nn.Linear(512, 512),
            #nn.ReLU(),
            nn.Linear(512, dim_out),
            #nn.ReLU()
        )

    def forward(self, x):
        logits = self.model_sequence(x)
        return logits
