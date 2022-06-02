import pickle

import src.midi_generator as midi_generator
import src.dataset_note_info_generator as note_generator

from src.dataset_one_hot_encoder import get_to_one_hot_encoding
from src.models.voices import voices

with open('./data/JSB Chorales.pickle', 'rb') as file:
    dataset = pickle.load(file)


train_hot_encodings = []
for song in dataset['train']:
    for voice in voices.values():
        one_hot_encoding = get_to_one_hot_encoding(song, voice)
        train_hot_encodings.append(one_hot_encoding)

# ==== Code to generate to midi. ====
train_song = dataset['train'][0]
track_note_infos = note_generator.generate_note_info(train_song)

midi_generator.generate_midi('file.mid', track_note_infos)
