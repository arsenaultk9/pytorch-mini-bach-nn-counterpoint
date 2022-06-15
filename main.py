
import src.midi_generator as midi_generator
import src.dataset_note_info_generator as note_generator

from src.data_loader import load_data

dataset = load_data()

# ==== Code to generate to midi. ====
train_song = dataset['train'][0]
track_note_infos = note_generator.generate_note_info(train_song)

midi_generator.generate_midi('file.mid', track_note_infos)
