import pickle
import src.dataset_note_info_generator as note_generator
import src.dataset_tuple_generator as tuple_generator
from src.data_augmentation.song_data_augmenter import augment_song_data

with open('./data/jsb-chorales-16th.pkl', 'rb') as file:
    midi_data = pickle.load(file, encoding="latin1")


song_tuple_datas = midi_data['train']
total_songs = len(song_tuple_datas)

song_tuple_datas_augmented = []

for index, song_tuple_data in enumerate(song_tuple_datas):
    print(f'Augmenting song {index + 1} of {total_songs} songs')

    voice_note_infos = note_generator.generate_note_info(song_tuple_data)
    song_augmentations = augment_song_data(voice_note_infos)

    for song_augmentation in song_augmentations:
        tuple_augmentation = tuple_generator.generate_tuple_form(song_augmentation)
        song_tuple_datas_augmented.append(tuple_augmentation)


midi_data['train'] = song_tuple_datas_augmented

# Store data (serialize)
with open('./data/scales_augmented.pkl', 'wb') as handle:
    pickle.dump(midi_data, handle, protocol=pickle.HIGHEST_PROTOCOL)