import torch
import torch.nn as F
import torch.optim as optim

import src.constants as constants

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NetworkTrainer:
    def __init__(self, network, data_loader):
        self.network = network
        self.data_loader = data_loader
        self.optimizer = optim.Adadelta(network.parameters(), lr=constants.OPTIMIZER_ADAM_LR)
        self.loss_function = F.CrossEntropyLoss()

    def get_instrument_loss(self, model_output, y_target):
        instrument_losses = 0

        for pos_index in range(constants.SEQUENCE_LENGTH):
            output_slice = model_output[:,pos_index]
            y_target_slice = y_target[:,pos_index] #.nonzero()[0][1].reshape(1, 1)

            # TODO: Use NLL Loss instead.

            current_loss = self.loss_function(output_slice, y_target_slice)
            instrument_losses += current_loss

        return instrument_losses

    def epoch_train(self, epoch):
        self.network.train()

        for batch_idx, (x_soprano, y_alto, y_tenor, y_bass) in enumerate(self.data_loader):
            x_soprano = x_soprano.to(device)
            y_alto = y_alto.to(device)
            y_tenor = y_tenor.to(device)
            y_bass = y_bass.to(device)

            self.optimizer.zero_grad()
            output_alto, output_tenor, output_bass = self.network(x_soprano)

            loss_alto = self.get_instrument_loss(output_alto, y_alto)
            loss_tenor = self.get_instrument_loss(output_tenor, y_tenor)
            loss_bass = self.get_instrument_loss(output_bass, y_bass)

            loss_total = loss_alto + loss_tenor + loss_bass

            loss_total.backward()
            self.optimizer.step()

            if batch_idx % constants.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x_soprano), len(self.data_loader.dataset),
                    100. * batch_idx / len(self.data_loader), loss_total.item()))

