# Maxime Darrin 2020
# Usual classification models for MNIST

import torch.nn as nn
import torch.nn.functional as F


class mnistConv(nn.Module):
    '''
    Convolutional model used for mnist experiments
    '''
    def __init__(self, device='cpu'):

        super(mnistConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5).to(device)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5).to(device)
        self.conv2_drop = nn.Dropout2d().to(device)
        self.fc1 = nn.Linear(320, 50).to(device)
        self.fc2 = nn.Linear(50, 10).to(device)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class mnistDense(nn.Module):
    '''
    Dense model used for mnist experiments
    '''
    def __init__(self, device='cpu'):
        super(mnistDense, self).__init__()

        self.seq = nn.Sequential(nn.Linear(28 * 28, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 10),
                                 nn.LogSoftmax(dim=1))

    def forward(self, x):
        batch_size = x.shape[0]

        return self.seq(x.view(batch_size, 28 * 28))
