from common import torch
from torch import nn

class CNNA(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.head = nn.Linear(64, k, bias=True)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.relu(self.conv3(x)).mean(dim=(2, 3))
        return self.head(x)
