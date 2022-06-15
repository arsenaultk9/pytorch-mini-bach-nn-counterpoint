import pickle


from src.dataset_one_hot_encoder import get_to_one_hot_encoding
from src.sequence_length_splitter import split_into_sequences
from src.models.voices import voices

def load_data():
    with open('./data/jsb-chorales-16th.pkl', 'rb') as file:
        dataset = pickle.load(file, encoding="latin1")

    sequence_data = dict()

    sequence_data['soprano'] = []
    sequence_data['alto'] = []
    sequence_data['tenor'] = []
    sequence_data['bass'] = []

    for song in dataset['train']:
        for voice in voices.values():
            one_hot_encoding = get_to_one_hot_encoding(song, voice)
            sequences_split = split_into_sequences(one_hot_encoding)

            sequence_data[voice.name] = sequence_data[voice.name] + sequences_split


    return dataset