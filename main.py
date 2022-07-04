import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import src.midi_generator as midi_generator
import src.dataset_note_info_generator as note_generator
import src.constants as constants

from src.data_loader import load_data
from src.network_trainer import NetworkTrainer
from src.networks.forward_network import ForwardNetwork

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

midi_data, dataset = load_data()

data_loader = DataLoader(dataset)
network = ForwardNetwork().to(device)
trainer = NetworkTrainer(network, data_loader)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)

# ==== Code to generate to midi. ====
train_song = midi_data['train'][0]
track_note_infos = note_generator.generate_note_info(train_song)

midi_generator.generate_midi('file.mid', track_note_infos)
