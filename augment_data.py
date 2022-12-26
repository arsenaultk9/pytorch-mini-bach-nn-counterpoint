import pickle
import src.constants as constants
import src.dataset_note_info_generator as note_generator
from src.data_augmentation.song_data_augmenter import augment_song_data

with open('./data/jsb-chorales-16th.pkl', 'rb') as file:
    midi_data = pickle.load(file, encoding="latin1")


song_tuple_datas = midi_data['train']
song_tuple_datas_augmented = []

song_note_infos = []
for index, song_tuple_data in enumerate(song_tuple_datas):
    voice_note_infos = note_generator.generate_note_info(song_tuple_data)
    data_augmented = augment_song_data(voice_note_infos)

    song_note_infos.append(voice_note_infos)


midi_data['train'] = song_tuple_datas_augmented

# Store data (serialize)
with open('./data/augmented.pkl', 'wb') as handle:
    pickle.dump(midi_data, handle, protocol=pickle.HIGHEST_PROTOCOL)