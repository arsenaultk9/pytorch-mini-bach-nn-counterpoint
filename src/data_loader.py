import pickle

from src.dataset_one_hot_encoder import get_to_one_hot_encoding
from src.sequence_length_splitter import split_into_sequences
from src.models.voices import voices
from src.models.chorales_dataset import ChoralesDataset


def get_dataset(song_data):
    sequence_data = dict()

    sequence_data['soprano'] = []
    sequence_data['alto'] = []
    sequence_data['tenor'] = []
    sequence_data['bass'] = []

    for song in song_data:
        for voice in voices.values():
            one_hot_encoding = get_to_one_hot_encoding(song, voice)
            sequences_split = split_into_sequences(one_hot_encoding)

            sequence_data[voice.name] = sequence_data[voice.name] + \
                sequences_split

    return ChoralesDataset(sequence_data)


def load_data():
    with open('./data/jsb-chorales-16th.pkl', 'rb') as file:
        midi_data = pickle.load(file, encoding="latin1")

    return midi_data, get_dataset(midi_data['train']), get_dataset(midi_data['test']), get_dataset(midi_data['valid'])
