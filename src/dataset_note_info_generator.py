from typing import List
from src.models.note_info import NoteInfo


def get_next_note_duration_and_note(dataset_song, track_number: int):
    cur_pos = 0
    cur_note_pos = 0
    cur_length = 0
    cur_note_number = dataset_song[0][track_number]

    for note_index in range(len(dataset_song)):
        current_notes_tuple = dataset_song[note_index]
        next_note_number = current_notes_tuple[track_number] if len(
            current_notes_tuple) > track_number else None

        if next_note_number != cur_note_number:
            yield (cur_note_pos, cur_length, cur_note_number)
            cur_length = 0
            cur_note_pos = cur_pos

        cur_note_number = next_note_number
        cur_length += 0.25
        cur_pos += 0.25


def get_track_notes(dataset_song, track_number: int) -> List[NoteInfo]:
    note_infos = []

    for pos, note_length, note_number in get_next_note_duration_and_note(dataset_song, track_number):

        if note_number == None:
            continue

        note_info = NoteInfo.create(pos, note_length, note_number)
        note_infos.append(note_info)

    return note_infos

# Data set example:
# (72, 76, 79, 84)
# (72, 76, 79, 84)
# (71, 74, 79, 86)
# (73, 77, 81, 88)

# Note: When a voice is missing, there are no None placeholder, so in the SATB voices the highest pitch are automatically penelized
# although the silence note could be in a lower voice which is hard to deduce.


def generate_note_info(dataset_song) -> List[List[NoteInfo]]:
    track_note_infos = []

    for track_number in range(4):
        note_infos = get_track_notes(dataset_song, track_number)
        track_note_infos.append(note_infos)

    return track_note_infos
