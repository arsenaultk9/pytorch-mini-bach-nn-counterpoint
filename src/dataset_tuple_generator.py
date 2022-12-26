from typing import List, Tuple
import numpy as np

from src.models.note_info import NoteInfo

def get_last_pos(voices_note_infos: List[List[NoteInfo]]) -> float:
    last_pos = 0

    for voice_notes in voices_note_infos:
        for note_info in voice_notes:
            note_end = note_info.starting_beat + note_info.length

            last_pos = max(last_pos, note_end)

    return last_pos

def get_notes_on_at_position(voices_note_infos: List[List[NoteInfo]], position: float) -> List[NoteInfo]:
    for voice_notes in voices_note_infos:
        for note_info in voice_notes:
            if not note_info.is_on_at_beat(position):
                continue

            yield note_info

def get_voice_tuples(voices_note_infos: List[List[NoteInfo]]) -> Tuple[int]:
    last_pos = get_last_pos(voices_note_infos)

    for cur_pos in np.arange(0, last_pos, 0.25):
        notes_on_pos = list(get_notes_on_at_position(voices_note_infos, cur_pos))
        note_numbers = tuple(map(lambda n: n.pitch, notes_on_pos))

        yield note_numbers


def generate_tuple_form(voices_note_infos: List[List[NoteInfo]]) -> List[Tuple[int]]:
    song_tuple_form = []

    for current_voice_tuple in get_voice_tuples(voices_note_infos):
        song_tuple_form.append(current_voice_tuple)

    return song_tuple_form