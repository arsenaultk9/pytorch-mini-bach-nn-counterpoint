import numpy as np
import torch
from torch.utils.data import Dataset


def encode_to_active_indexes(sequence_data):
    on_sequence_indexes = []

    for sequence in sequence_data:
        sequence_slice_indexes = []

        for sequence_slice in sequence:
            on_index = np.where(sequence_slice == 1.0)[0][0]
            sequence_slice_indexes.append(on_index)

        on_sequence_indexes.append(sequence_slice_indexes)

    return on_sequence_indexes


class ChoralesDataset(Dataset):
    def __init__(self, sequences_data: dict):
        self.X_soprano = torch.tensor(
            sequences_data['soprano'], dtype=torch.float32)

        self.Y_alto = torch.tensor(encode_to_active_indexes(
            sequences_data['alto']), dtype=torch.long)
        self.Y_tenor = torch.tensor(encode_to_active_indexes(
            sequences_data['tenor']), dtype=torch.long)
        self.Y_bass = torch.tensor(encode_to_active_indexes(
            sequences_data['bass']), dtype=torch.long)

        self.length = len(sequences_data['soprano'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X_soprano_formatted = self.X_soprano[idx]
        Y_alto_formatted = self.Y_alto[idx]
        Y_tenor_formatted = self.Y_tenor[idx]
        Y_bass = self.Y_bass[idx]

        return (X_soprano_formatted, Y_alto_formatted, Y_tenor_formatted, Y_bass)
