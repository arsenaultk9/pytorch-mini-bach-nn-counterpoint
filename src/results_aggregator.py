from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ResultsAggregator:
    def __init__(self):
        self.total_losses = []
        self.total_right_predictions = []

    def aggregate_loss(self, current_batch_loss):
        self.total_losses.append(current_batch_loss.item())

    def aggregate_right_predictions(self, sample_right_predictions):
        self.total_right_predictions.append(sample_right_predictions)

    def get_average_loss(self, current_train_item):
        total_loss = sum(self.total_losses)
        return total_loss / (current_train_item + 1)

    def get_average_right_predictions(self, sample_count):
        total_right_predictions = sum(self.total_right_predictions)
        average_right_predictions = total_right_predictions / sample_count

        return f"{average_right_predictions:.1f}"


    def update_plots(self, mode, current_train_item, sample_count, epoch):
        average_loss = self.get_average_loss(current_train_item)
        average_right_predictions = self.get_average_loss(sample_count)

        writer.add_scalar(f'Loss/{mode}', average_loss, epoch)
        writer.add_scalar(f'Accuracy/{mode}', average_right_predictions, epoch)

