import sys
import src.constants as constants
import numpy as np


def generate_song_matrix(note_infos):
    min_beat_pos = sys.maxsize
    max_beat_pos = 0

    for note_info in note_infos:
        if(note_info.starting_beat <= min_beat_pos):
            min_beat_pos = note_info.starting_beat

        note_end = note_info.starting_beat + note_info.length
        if(note_end >= max_beat_pos):
            max_beat_pos = note_end

    song_length = int((max_beat_pos - min_beat_pos + 1)
                      * constants.SEGMENTS_PER_BEAT)
    song_matrix = np.zeros(
        (constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED, song_length))

    # Not the most optimized way to do this but at least it's simple so easier to debug :)
    for song_beat_pos in range(song_length):
        current_song_pos_quadrant = min_beat_pos + \
            song_beat_pos / constants.SEGMENTS_PER_BEAT

        has_note = False
        for note_info in note_infos:
            if not note_info.is_on_at_beat(current_song_pos_quadrant):
                continue

            if note_info.pitch < constants.BOTTOM_SKIPPING_INDEX:
                continue

            if note_info.pitch > constants.TOP_SKIPPING_INDEX:
                continue

            song_matrix[note_info.pitch + 1 -
                        constants.BOTTOM_SKIPPING_INDEX, song_beat_pos] = 1
            has_note = True

        if not has_note:
            song_matrix[constants.SILENCE_INDEX, song_beat_pos] = 1

    return min_beat_pos, max_beat_pos, np.swapaxes(song_matrix, 0, 1)
