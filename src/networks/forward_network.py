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
        self.forward_tenor = nn.Linear(200, constants.SEQUENCE_LENGTH * 22)
        self.forward_bass = nn.Linear(200, constants.SEQUENCE_LENGTH * 29)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.input(x)
        x = self.hidden(x)
        x = F.relu(x)

        # alto
        x_alto = self.forward_alto(x)
        x_alto = F.relu(x_alto)
        x_alto = torch.reshape(x_alto, (1, constants.SEQUENCE_LENGTH, 22))
        y_alto = F.softmax(x_alto, dim=2) # Calculate for each part of sequence & not on whole sequence
        # TODO: Use log_softmax instead?

        # tenor
        x_tenor = self.forward_tenor(x)
        x_tenor = F.relu(x_tenor)
        x_tenor = torch.reshape(x_tenor, (1, constants.SEQUENCE_LENGTH, 22))
        y_tenor = F.softmax(x_tenor, dim=2) # Calculate for each part of sequence & not on whole sequence
        # TODO: Use log_softmax instead?

        # bass
        x_bass = self.forward_bass(x)
        x_bass = F.relu(x_bass)
        x_bass = torch.reshape(x_bass, (1, constants.SEQUENCE_LENGTH, 29))
        y_bass = F.softmax(x_bass, dim=2) # Calculate for each part of sequence & not on whole sequence
        # TODO: Use log_softmax instead?

        return y_alto, y_tenor, y_bass