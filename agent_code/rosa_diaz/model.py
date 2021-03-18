import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class DQN(nn.Module):
    gamma = 1.0
    learning_rate = 0.0003

    def __init__(self, dim_in, dim_out):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(dim_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, dim_out),
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        #self.to(self.device)

    def forward(self, x):
        #x.to(self.device)
        logits = self.model_sequence(x)
        return logits
