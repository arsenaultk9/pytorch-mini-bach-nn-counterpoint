import torch
import torch.nn as F
import torch.optim as optim
from torch.utils.data import DataLoader

import src.constants as constants
import src.metrics as metrics
from src.networks.forward_network import ForwardNetwork
from src.results_aggregator import ResultsAggregator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class NetworkTrainer:
    def __init__(self,
                 network: ForwardNetwork,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 valid_data_loader: DataLoader):

        self.network = network
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.valid_data_loader = valid_data_loader

        self.loss_function = F.NLLLoss()
        self.optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)

    def get_instrument_loss(self, model_output, y_target):
        return self.loss_function(model_output, y_target)

    def epoch_train(self, epoch):
        self.network.train()

        results_aggregator = ResultsAggregator()
        result_agg_sample_size = 0

        for batch_idx, (x_soprano, y_alto, y_tenor, y_bass) in enumerate(self.train_data_loader):
            if len(x_soprano) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            self.optimizer.zero_grad()

            x_soprano = x_soprano.to(device)
            y_alto = y_alto.to(device)
            y_tenor = y_tenor.to(device)
            y_bass = y_bass.to(device)

            output_alto, output_tenor, output_bass = self.network(x_soprano)

            loss_alto = self.get_instrument_loss(output_alto, y_alto)
            loss_tenor = self.get_instrument_loss(output_tenor, y_tenor)
            loss_bass = self.get_instrument_loss(output_bass, y_bass)

            current_batch_loss = loss_alto + loss_tenor + loss_bass
            results_aggregator.aggregate_loss(current_batch_loss)

            current_batch_loss.backward()
            self.optimizer.step()

            if batch_idx % constants.BATCH_LOG_INTERVAL == 0 and batch_idx != 0:
                current_item = batch_idx * len(x_soprano)
                result_agg_sample_size += len(x_soprano)

                # Only sample at log because of expensive calculation
                sample_right_predictions = metrics.get_batch_right_predictions([output_alto, output_tenor, output_bass], [y_alto, y_tenor, y_bass])
                results_aggregator.aggregate_right_predictions(sample_right_predictions)

                sample_note_accuracy = metrics.get_batch_note_accuracy([output_alto, output_tenor, output_bass], [y_alto, y_tenor, y_bass])
                results_aggregator.aggregate_note_accuracy(sample_note_accuracy)

                average_right_predictions = results_aggregator.get_average_right_predictions(result_agg_sample_size)
                average_note_accuracy = results_aggregator.get_average_note_accuracy(result_agg_sample_size)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tAverage Loss: {:.6f}\tAverage Right Predictions: {}\tAverage Note Accuracy: {}'.format(
                    f"{epoch:03d}",
                    f"{current_item:04d}",
                    len(self.train_data_loader.dataset),
                    100. * batch_idx / len(self.train_data_loader),
                    results_aggregator.get_average_loss(current_item),
                    f"{average_right_predictions:.1f}",
                    f"{average_note_accuracy:.1f}"))

        results_aggregator.update_plots('train', len(self.train_data_loader.dataset), result_agg_sample_size, epoch)

    def epoch_valid(self, epoch):
        self.network.eval()

        results_aggregator = ResultsAggregator()
        result_agg_sample_size = 0

        for batch_idx, (x_soprano, y_alto, y_tenor, y_bass) in enumerate(self.valid_data_loader):
            if len(x_soprano) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            x_soprano = x_soprano.to(device)
            y_alto = y_alto.to(device)
            y_tenor = y_tenor.to(device)
            y_bass = y_bass.to(device)

            output_alto, output_tenor, output_bass = self.network(x_soprano)

            loss_alto = self.get_instrument_loss(output_alto, y_alto)
            loss_tenor = self.get_instrument_loss(output_tenor, y_tenor)
            loss_bass = self.get_instrument_loss(output_bass, y_bass)

            current_batch_loss = loss_alto + loss_tenor + loss_bass
            results_aggregator.aggregate_loss(current_batch_loss)

            if batch_idx % constants.VALID_PREDICTION_SAMPLE_RATE == 0 and batch_idx != 0:
                result_agg_sample_size += len(x_soprano)

                # Only sample at log because of expensive calculation
                sample_right_predictions = metrics.get_batch_right_predictions([output_alto, output_tenor, output_bass], [y_alto, y_tenor, y_bass])
                results_aggregator.aggregate_right_predictions(sample_right_predictions)

                sample_note_accuracy = metrics.get_batch_note_accuracy([output_alto, output_tenor, output_bass], [y_alto, y_tenor, y_bass])
                results_aggregator.aggregate_note_accuracy(sample_note_accuracy)
        
        average_loss = results_aggregator.get_average_loss(len(self.valid_data_loader.dataset))

        if constants.APPLY_LR_SCHEDULER:
            self.lr_scheduler.step(average_loss)
            learning_rate = self.lr_scheduler.optimizer.param_groups[0]['lr']
            print(f'learning_rate: {learning_rate}')

        average_right_predictions = results_aggregator.get_average_right_predictions(result_agg_sample_size)
        average_note_accuracy = results_aggregator.get_average_note_accuracy(result_agg_sample_size)

        results_aggregator.update_plots('valid', len(self.valid_data_loader.dataset), result_agg_sample_size, epoch)

        print('Valid Epoch: {}\tAverage Loss: {:.6f}\tAverage Right Predictions: {}\tAverage Note Accuracy: {}'.format(
            f"{epoch:03d}",
            average_loss,
            f"{average_right_predictions:.1f}",
            f"{average_note_accuracy:.1f}"))


    def test(self):
        self.network.eval()

        results_aggregator = ResultsAggregator()

        for _, (x_soprano, y_alto, y_tenor, y_bass) in enumerate(self.test_data_loader):
            if len(x_soprano) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            x_soprano = x_soprano.to(device)
            y_alto = y_alto.to(device)
            y_tenor = y_tenor.to(device)
            y_bass = y_bass.to(device)

            output_alto, output_tenor, output_bass = self.network(x_soprano)

            loss_alto = self.get_instrument_loss(output_alto, y_alto)
            loss_tenor = self.get_instrument_loss(output_tenor, y_tenor)
            loss_bass = self.get_instrument_loss(output_bass, y_bass)

            current_batch_loss = loss_alto + loss_tenor + loss_bass
            results_aggregator.aggregate_loss(current_batch_loss)

            sample_right_predictions = metrics.get_batch_right_predictions([output_alto, output_tenor, output_bass], [y_alto, y_tenor, y_bass])
            results_aggregator.aggregate_right_predictions(sample_right_predictions)

            sample_note_accuracy = metrics.get_batch_note_accuracy([output_alto, output_tenor, output_bass], [y_alto, y_tenor, y_bass])
            results_aggregator.aggregate_note_accuracy(sample_note_accuracy)

        test_item_count = len(self.test_data_loader.dataset)

        average_loss = results_aggregator.get_average_loss(test_item_count)
        average_right_predictions = results_aggregator.get_average_right_predictions(test_item_count)
        average_note_accuracy = results_aggregator.get_average_note_accuracy(test_item_count)

        print('Test Epoch: Average Loss: {:.6f}\tAverage Right Predictions: {}\tAverage Note Accuracy: {}'.format(
            average_loss,
            f"{average_right_predictions:.1f}",
            f"{average_note_accuracy:.1f}"))
