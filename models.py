import torch
from torch import nn


def noise(size):
    n = torch.Tensor(torch.randn((size, 100)))

    return n


def hidden_layer(input, output, dropout=False):
    if dropout:
        return nn.Sequential(
            nn.Linear(input, output),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
    else:
        return nn.Sequential(
            nn.Linear(input, output),
            nn.LeakyReLU(0.2)
        )


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = hidden_layer(n_features, 1024, dropout=True)

        self.hidden1 = hidden_layer(1024, 512, dropout=True)

        self.hidden2 = hidden_layer(512, 256, dropout=True)

        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)

        return x


class GeneratorNetwork(nn.Module):

    def __init__(self):

        super(GeneratorNetwork, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = hidden_layer(n_features, 256)

        self.hidden1 = hidden_layer(256, 512)

        self.hidden2 = hidden_layer(512, 1024)

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)

        return x