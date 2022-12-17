import torch
import src.constants as constants

def get_sample_right_predictions(outputs, y_targets):
    sample_right_predictions = 0

    for index, _ in enumerate(outputs):
        output = outputs[index]
        y = y_targets[index]

        sample_right_predictions += get_total_right_predictions(output, y)

    return sample_right_predictions

def get_total_right_predictions(model_output, y_target):
    total_right_predictions = 0

    model_transposed = model_output[0].transpose(0, 1)

    for slice_index in range(constants.SEQUENCE_LENGTH):
        model_prediction = torch.argmax(model_transposed[slice_index])
        target_prediction = y_target[0][slice_index]

        if model_prediction == target_prediction:
            total_right_predictions += 1

    return total_right_predictions


def get_sample_note_accuracy(outputs, y_targets):
    sample_right_predictions = 0

    for index, _ in enumerate(outputs):
        output = outputs[index]
        y = y_targets[index]

        sample_right_predictions += get_total_note_accuracy(output, y)

    return sample_right_predictions

def get_total_note_accuracy(model_output, y_target):
    total_right_predictions = 0

    model_transposed = model_output[0].transpose(0, 1)

    for slice_index in range(constants.SEQUENCE_LENGTH):
        model_prediction = torch.argmax(model_transposed[slice_index])
        target_prediction = y_target[0][slice_index]

        if model_prediction == target_prediction:
            total_right_predictions += 1

    return total_right_predictions