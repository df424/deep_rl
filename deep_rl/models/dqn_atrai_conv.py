
import torch

class AtariDQNQNet(torch.nn.Module):

    def __init__(self, num_actions: int):
        super(AtariDQNQNet, self).__init__()
        self._num_actions = num_actions

        self._conv1 = torch.nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self._relu = torch.nn.ReLU(False)
        self._conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self._linear1 = torch.nn.Linear(in_features=2592, out_features=256)
        self._linear2 = torch.nn.Linear(in_features=256, out_features=self._num_actions)

    def forward(self, inputs):
        x = self._relu.forward(self._conv1.forward(inputs))
        x = self._relu.forward(self._conv2.forward(x))
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = self._relu.forward(self._linear1(x))
        x = self._linear2(x)
        return x