import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)