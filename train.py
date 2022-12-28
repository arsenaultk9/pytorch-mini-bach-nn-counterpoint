import torch
import random
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

midi_data, train_dataset, test_dataset, valid_dataset = load_data()

train_data_loader = DataLoader(train_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
test_data_loader = DataLoader(test_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
valid_data_loader = DataLoader(valid_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

network = ForwardNetwork().to(device)
trainer = NetworkTrainer(network, train_data_loader, test_data_loader, valid_data_loader)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)
    trainer.epoch_valid(epoch)

trainer.test()

# Turn off training mode & switch to model evaluation
network.eval()

# === Save model for production use ===
(x_soprano_sample, y_alto, y_tenor, y_bass) = train_dataset[0:constants.BATCH_SIZE]
traced_script_module = torch.jit.trace(network.forward, x_soprano_sample.to(device))
traced_script_module.save("result_model/satb_forward_network.pt")

# ==== Code to generate to midi. ====
random_start_seed = random.randrange(0, len(test_dataset) - constants.BATCH_SIZE)

for song_index in range(random_start_seed, random_start_seed + 9):
    file_index = song_index - random_start_seed + 1
    print(f'Generating song {file_index}')

    harmony_generator = NetworkHarmonyGenerator(network)
    (x_soprano_sample, y_alto, y_tenor, y_bass) = test_dataset[song_index:song_index+constants.BATCH_SIZE]

    generated_song = harmony_generator.generate_harmony(x_soprano_sample)
    original_song = harmony_generator.imitate_harmony(
        x_soprano_sample[0], y_alto[0], y_tenor[0], y_bass[0])

    generated_note_infos = note_generator.generate_note_info(generated_song)
    original_note_infos = note_generator.generate_note_info(original_song)
    melody_note_infos = note_generator.generate_note_info_melody(original_song)

    midi_generator.generate_midi(f'generated_file{file_index}.mid', generated_note_infos)
    midi_generator.generate_midi(f'original_file{file_index}.mid', original_note_infos)
    midi_generator.generate_midi(f'melody_file{file_index}.mid', melody_note_infos)

