import torch
from torch.utils.data import Dataset


class ChoralesDataset(Dataset):
    def __init__(self, sequence_data: dict):
        self.X_soprano = torch.tensor(sequence_data['soprano'])
        self.Y_alto = torch.tensor(sequence_data['alto'])
        self.Y_tenor = torch.tensor(sequence_data['tenor'])
        self.Y_bass = torch.tensor(sequence_data['bass'])

        self.length = len(sequence_data['soprano'])

    def __len__(self):
        self.length

    def __getitem__(self, idx):
        return self.X_soprano[idx], self.Y_alto[idx], self.Y_tenor[idx], self.Y_bass[idx]
