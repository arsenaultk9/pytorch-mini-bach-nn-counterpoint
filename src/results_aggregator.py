from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ResultsAggregator:
    def __init__(self):
        self.total_losses = []
        self.total_right_predictions = []
        self.total_note_accuracy = []

    def aggregate_loss(self, current_batch_loss):
        self.total_losses.append(current_batch_loss.item())

    def aggregate_right_predictions(self, sample_right_predictions):
        self.total_right_predictions.append(sample_right_predictions)

    def aggregate_note_accuracy(self, sample_note_accuracy):
        self.total_note_accuracy.append(sample_note_accuracy)

    def get_average_loss(self, item_count):
        total_loss = sum(self.total_losses)
        return total_loss / (item_count + 1)

    def get_average_right_predictions(self, item_count):
        total_right_predictions = sum(self.total_right_predictions)
        average_right_predictions = total_right_predictions / item_count

        return average_right_predictions

    def get_average_note_accuracy(self, item_count):
        total_note_accuracy = sum(self.total_note_accuracy)
        average_note_accuracy = total_note_accuracy / item_count

        return average_note_accuracy


    def update_plots(self, mode, item_count, result_agg_sample_size, epoch):
        average_loss = self.get_average_loss(item_count)
        average_right_predictions = self.get_average_right_predictions(result_agg_sample_size)
        average_note_accuracy = self.get_average_note_accuracy(result_agg_sample_size)

        writer.add_scalar(f'Loss/{mode}', average_loss, epoch)
        writer.add_scalar(f'Accuracy/{mode}', average_right_predictions, epoch)
        writer.add_scalar(f'Note Accuracy/{mode}', average_note_accuracy, epoch)

