import torch
import src.constants as constants

def get_batch_right_predictions(outputs, y_targets):
    batch_right_predictions = 0

    for index, _ in enumerate(outputs):
        output = outputs[index]
        y = y_targets[index]

        batch_right_predictions += get_total_right_predictions(output, y)

    return batch_right_predictions

def get_total_right_predictions(model_output, y_target):
    total_right_predictions = 0

    for batch_index in range(constants.BATCH_SIZE):
        model_transposed = model_output[batch_index].transpose(0, 1)

        for slice_index in range(constants.SEQUENCE_LENGTH):
            model_prediction = torch.argmax(model_transposed[slice_index])
            target_prediction = y_target[batch_index][slice_index]

            if model_prediction == target_prediction:
                total_right_predictions += 1

    return total_right_predictions


def get_batch_note_accuracy(outputs, y_targets):
    sample_right_predictions = 0

    for index, _ in enumerate(outputs):
        output = outputs[index]
        y = y_targets[index]

        sample_right_predictions += get_total_note_accuracy(output, y)

    return sample_right_predictions

def get_total_note_accuracy(model_output, y_target):
    total_right_predictions = 0

    for batch_index in range(constants.BATCH_SIZE):
        model_transposed = model_output[batch_index].transpose(0, 1)

        for measure_index in range(constants.MEASURES):
            measure_start = measure_index * constants.MEASURE_LENGTH
            measure_end = (measure_index + 1) * constants.MEASURE_LENGTH
            measure_target_notes = y_target[batch_index][measure_start:measure_end]

            for sequence_index in range(measure_start, measure_end):
                model_prediction = torch.argmax(model_transposed[sequence_index])
                
                if (model_prediction.item() in measure_target_notes):
                    total_right_predictions += 1

    return total_right_predictions