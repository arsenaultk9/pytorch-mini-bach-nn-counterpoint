import numpy.typing as npt

from src import constants


def split_into_sequences(one_hot_encoding: npt.ArrayLike):
    sequences = []

    seq_length = one_hot_encoding.shape[0]

    for seq_start in range(0, seq_length, constants.SEQUENCE_LENGTH):
        seq_end = seq_start + constants.SEQUENCE_LENGTH

        if seq_end > seq_length:
            seq_end = seq_length - 1
            seq_start = seq_end - constants.SEQUENCE_LENGTH

        sequence = one_hot_encoding[seq_start:seq_end]
        sequences.append(sequence)

    return sequences