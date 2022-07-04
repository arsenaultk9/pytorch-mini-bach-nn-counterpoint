import torch
import torch.nn.functional as F
import torch.optim as optim

import src.constants as constants

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NetworkTrainer:
    def __init__(self, network, data_loader):
        self.network = network
        self.data_loader = data_loader
        self.optimizer = optim.Adadelta(network.parameters(), lr=constants.OPTIMIZER_ADAM_LR)

    def epoch_train(self, epoch):
        self.network.train()

        for batch_idx, (x_soprano, y_alto, y_tenor, y_bass) in enumerate(self.data_loader):
            x_soprano = x_soprano.to(device)
            y_alto = y_alto.to(device)
            y_tenor = y_tenor.to(device)
            y_bass = y_bass.to(device)


            self.optimizer.zero_grad()
            output_alto, output_tenor, output_bass = self.network(x_soprano)

            raise Exception('Must continue this code')
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % constants.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))

