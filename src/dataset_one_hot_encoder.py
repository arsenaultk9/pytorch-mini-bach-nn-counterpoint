import numpy as np
from src import constants
from src.models.voice import Voice


def get_to_one_hot_encoding(song: list, voice: Voice):
    one_hot_encoding = np.zeros((len(song), voice.range.range_and_silence_length()))

    for song_position, song_segment in enumerate(song):
        voice_note = voice.get_note_in_segment(song_segment)

        if voice_note == -1:
            one_hot_encoding[song_position][constants.SILENCE_INDEX] = 1
            continue

        hot_encoding_note_pos = voice_note - voice.range.min_note + 1
        one_hot_encoding[song_position][hot_encoding_note_pos] = 1