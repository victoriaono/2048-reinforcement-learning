import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N1_FILTERS = 512
N2_FILTERS = 4096

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, (1, 2), padding='valid')
        self.conv2 = nn.Conv2d(input_dim, output_dim, (2, 1), padding='valid')

    def forward(self, x):
        x = x.to(device)
        output1 = F.relu(self.conv1(x))
        output2 = F.relu(self.conv2(x))
        return output1, output2

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        # Layer 1
        self.conv1 = ConvBlock(16, N1_FILTERS)
        # Layer 2
        self.conv2 = ConvBlock(N1_FILTERS, N2_FILTERS)
        # Concatenate and fully connect
        hidden1 = N1_FILTERS * 4 * 3
        hidden12 = N2_FILTERS * 3 * 3
        hidden21 = N2_FILTERS * 2 * 4
        self.dense = nn.Linear(2 * hidden1 + 2 * hidden21 + 2 * hidden12, 4)
    
    def forward(self, x):
        x = x.to(device)
        x1, x2 = self.conv1(x)
        x11, x12 = self.conv2(x1)
        x21, x22 = self.conv2(x2)
        x = torch.cat([nn.Flatten()(x) for x in [x1, x2, x11, x12, x21, x22]], dim=1)
        return self.dense(x)