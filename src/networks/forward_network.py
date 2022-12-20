import torch
import torch.nn as nn
import torch.nn.functional as F

import src.constants as constants


class ForwardNetwork(nn.Module):
    def __init__(self):
        super(ForwardNetwork, self).__init__()

        self.input = nn.Linear(constants.SEQUENCE_LENGTH * 22, 200)
        self.hidden1 = nn.Linear(200, 200)
        self.dropout1 = nn.Dropout(0.5)
        self.hidden2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5)

        self.forward_alto = nn.Linear(200, constants.SEQUENCE_LENGTH * 22)
        self.forward_tenor = nn.Linear(200, constants.SEQUENCE_LENGTH * 22)
        self.forward_bass = nn.Linear(200, constants.SEQUENCE_LENGTH * 29)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.input(x)

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # alto
        x_alto = self.forward_alto(x)
        x_alto = F.relu(x_alto)
        x_alto = torch.reshape(x_alto, (constants.BATCH_SIZE, 22, constants.SEQUENCE_LENGTH))
        y_alto = F.log_softmax(x_alto, dim=1) # Calculate for each part of sequence & not on whole sequence

        # tenor
        x_tenor = self.forward_tenor(x)
        x_tenor = F.relu(x_tenor)
        x_tenor = torch.reshape(x_tenor, (constants.BATCH_SIZE, 22, constants.SEQUENCE_LENGTH))
        y_tenor = F.log_softmax(x_tenor, dim=1) # Calculate for each part of sequence & not on whole sequence

        # bass
        x_bass = self.forward_bass(x)
        x_bass = F.relu(x_bass)
        x_bass = torch.reshape(x_bass, (constants.BATCH_SIZE, 29, constants.SEQUENCE_LENGTH))
        y_bass = F.log_softmax(x_bass, dim=1) # Calculate for each part of sequence & not on whole sequence

        return y_alto, y_tenor, y_bass