import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

class ResidualBlock(nn.Module):
    '''
    Used in the IMPALA blocks.
    '''
    def __init__(self, in_channels):
        super().__init__() # <-
        self.convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.convolution2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.convolution1(self.relu(x))
        output = self.convolution2(self.relu(output))
        return output + x

class ImpalaBlock(nn.Module):
    '''
    Used in the IMPALA encoder. The IMPALA encoder will use three of these
    blocks with varying in- and out-channels.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(out_channels)
        self.residual_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.convolution(x)
        x = self.maxpool(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        return x

class ImpalaEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features = feature_dim)
        self.relu = nn.ReLU()
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

class ImpalaEncoder_2(nn.Module):
    '''
    Used to test the expanded version of the impala encoder where
    the channels have been scaled by a factor of 2 compared to the standard. 
    '''
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=32)
        self.block2 = ImpalaBlock(in_channels=32, out_channels=64)
        self.block3 = ImpalaBlock(in_channels=64, out_channels=64)
        self.fc = nn.Linear(in_features = 64 * 8 * 8, out_features = feature_dim)
        self.relu = nn.ReLU()
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

class ImpalaEncoder_3(nn.Module):
    '''
    Used to test the expanded version of the impala encoder where
    the channels have been scaled by a factor of 2 compared to the standard
    and an additional fully connected layer was added. 
    '''
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=32)
        self.block2 = ImpalaBlock(in_channels=32, out_channels=64)
        self.block3 = ImpalaBlock(in_channels=64, out_channels=64)
        self.fc1 = nn.Linear(in_features = 64 * 8 * 8, out_features = 2048)
        self.fc2 = nn.Linear(in_features = 2048, out_features = feature_dim)
        self.relu = nn.ReLU()
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

def f(x):
    '''
    Used to calculate the number of output features from each IMPALA block.
    '''
    x = math.sqrt(x)
    x = math.ceil(x/2)
    return x**2