from torch.utils.data import DataLoader

import src.midi_generator as midi_generator
import src.dataset_note_info_generator as note_generator

from src.data_loader import load_data

midi_data, dataset = load_data()
x_soprano, y_alto, y_tenor, y_bass = dataset[0]

print(x_soprano.shape)
print(y_alto.shape)
print(y_tenor.shape)
print(y_bass.shape)

# ==== Code to generate to midi. ====
train_song = midi_data['train'][0]
track_note_infos = note_generator.generate_note_info(train_song)

midi_generator.generate_midi('file.mid', track_note_infos)
