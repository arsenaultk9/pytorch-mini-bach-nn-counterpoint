import torch
import torch.nn as nn
import torch.nn.functional as F

import src.constants as constants


class ForwardNetwork(nn.Module):
    def __init__(self):
        super(ForwardNetwork, self).__init__()

        self.input = nn.Linear(constants.SEQUENCE_LENGTH * 22, 200)
        self.hidden = nn.Linear(200, 200)

        self.forward_alto = nn.Linear(200, constants.SEQUENCE_LENGTH * 22)
        # TODO: Reshape this as to be two dimensions and repeat for other output.

    def forward(self, x):
        print(x)
        x = torch.flatten(x, start_dim=1)

        print(x)
        x = self.input(x)
        x = self.hidden(x)
        x = F.relu(x)

        return x

        # TODO: Complete this. 1d or 2d input????

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.forward1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.forward2(x)

        output = F.log_softmax(x, dim=1)
        return output