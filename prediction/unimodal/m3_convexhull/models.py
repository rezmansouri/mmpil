import torch.nn as nn


class Model_1(nn.Module):
    '''empty'''

    def __init__(self, input_dim=128, in_channels=1, out_channels=4, latent_dim=128):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, 3, padding='same')
        self.flatten_dim = (input_dim // 4) ** 2 * out_channels * 2
        self.fc1 = nn.Linear(self.flatten_dim, self.flatten_dim//4)
        self.fc2 = nn.Linear(self.flatten_dim//4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''empty'''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class Model_2(nn.Module):
    '''empty'''

    def __init__(self, input_dim=128, in_channels=1, out_channels=8, latent_dim=128):
        super(Model_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, 3, padding='same')
        self.conv3 = nn.Conv2d(
            out_channels * 2, out_channels * 4, 3, padding='same')
        self.flatten_dim = (input_dim // 8) ** 2 * out_channels * 4
        self.fc1 = nn.Linear(self.flatten_dim, self.flatten_dim//2)
        self.fc2 = nn.Linear(self.flatten_dim // 2, self.flatten_dim//4)
        self.fc3 = nn.Linear(self.flatten_dim//4, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim//2)
        self.fc5 = nn.Linear(latent_dim//2, 1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''empty'''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max(x)
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


class Model_3(nn.Module):
    '''empty'''

    def __init__(self, input_dim=128, in_channels=1, out_channels=8):
        super(Model_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, 3, padding='same')
        self.conv3 = nn.Conv2d(
            out_channels * 2, out_channels * 4, 3, padding='same')
        self.flatten_dim = (input_dim // 8) ** 2 * out_channels * 4
        self.fc1 = nn.Linear(self.flatten_dim, self.flatten_dim//2)
        self.fc2 = nn.Linear(self.flatten_dim // 2, 1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''empty'''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max(x)
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
