import torch
import torch.nn as nn
import torch.nn.functional as F

import src.constants as constants
from src.models.voices import voices


class ForwardNetwork(nn.Module):
    def __init__(self):
        super(ForwardNetwork, self).__init__()

        self.input = nn.Linear(constants.SEQUENCE_LENGTH *
                               voices['soprano'].range_and_silence_length(), 200)
        self.hidden1 = nn.Linear(200, 200)
        self.dropout1 = nn.Dropout(0.5)
        self.hidden2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5)

        self.forward_alto = nn.Linear(
            200, constants.SEQUENCE_LENGTH * voices['alto'].range_and_silence_length())
        self.forward_tenor = nn.Linear(
            200, constants.SEQUENCE_LENGTH * voices['tenor'].range_and_silence_length())
        self.forward_bass = nn.Linear(
            200, constants.SEQUENCE_LENGTH * voices['bass'].range_and_silence_length())

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
        x_alto = torch.reshape(
            x_alto, (constants.BATCH_SIZE, voices['alto'].range_and_silence_length(), constants.SEQUENCE_LENGTH))
        # Calculate for each part of sequence & not on whole sequence
        y_alto = F.log_softmax(x_alto, dim=1)

        # tenor
        x_tenor = self.forward_tenor(x)
        x_tenor = F.relu(x_tenor)
        x_tenor = torch.reshape(
            x_tenor, (constants.BATCH_SIZE, voices['tenor'].range_and_silence_length(), constants.SEQUENCE_LENGTH))
        # Calculate for each part of sequence & not on whole sequence
        y_tenor = F.log_softmax(x_tenor, dim=1)

        # bass
        x_bass = self.forward_bass(x)
        x_bass = F.relu(x_bass)
        x_bass = torch.reshape(
            x_bass, (constants.BATCH_SIZE, voices['bass'].range_and_silence_length(), constants.SEQUENCE_LENGTH))
        # Calculate for each part of sequence & not on whole sequence
        y_bass = F.log_softmax(x_bass, dim=1)

        return y_alto, y_tenor, y_bass
