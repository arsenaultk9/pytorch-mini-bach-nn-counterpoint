import torch
from torch.utils.data import DataLoader

import src.midi_generator as midi_generator
import src.dataset_note_info_generator as note_generator
import src.constants as constants

from src.data_loader import load_data
from src.network_trainer import NetworkTrainer
from src.networks.forward_network import ForwardNetwork
from src.network_harmony_generator import NetworkHarmonyGenerator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

midi_data, dataset = load_data()

data_loader = DataLoader(dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
network = ForwardNetwork().to(device)
trainer = NetworkTrainer(network, data_loader)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)

# Turn off training mode & switch to model evaluation
network.eval()

# ==== Code to generate to midi. ====
for song_index in range(9):
    print(f'Generating song {song_index + 1}')

    harmony_generator = NetworkHarmonyGenerator(network)
    (x_soprano_sample, y_alto, y_tenor, y_bass) = dataset[song_index:song_index+constants.BATCH_SIZE]

    generated_song = harmony_generator.generate_harmony(x_soprano_sample)
    original_song = harmony_generator.imitate_harmony(
        x_soprano_sample[0], y_alto[0], y_tenor[0], y_bass[0])

    generated_note_infos = note_generator.generate_note_info(generated_song)
    original_note_infos = note_generator.generate_note_info(original_song)
    melody_note_infos = note_generator.generate_note_info_melody(original_song)

    midi_generator.generate_midi(f'generated_file{song_index + 1}.mid', generated_note_infos)
    midi_generator.generate_midi(f'original_file{song_index + 1}.mid', original_note_infos)
    midi_generator.generate_midi(f'melody_file{song_index + 1}.mid', melody_note_infos)
