import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)