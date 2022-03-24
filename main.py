import pickle

import src.midi_generator as midi_generator
import src.dataset_note_info_generator as note_generator

with open('./data/JSB Chorales.pickle', 'rb') as file:
    dataset = pickle.load(file)

test_song = dataset['test'][0]
track_note_infos = note_generator.generate_note_info(test_song)

midi_generator.generate_midi('file.mid', track_note_infos)
