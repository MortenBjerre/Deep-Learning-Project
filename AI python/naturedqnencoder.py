import torch
import torch.nn as nn
from utils import orthogonal_init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class NatureEncoder(nn.Module):
    '''
    NatureDQN encoder
    '''
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        #input : 64 x 64 x 3 (pixel x pixel x rgb)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
            nn.ReLU(), # 15 x 15 x 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
            nn.ReLU(), # 6 x 6 x 64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(), # 4 x 4 x 64 = 1024
            Flatten(),
            nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)
