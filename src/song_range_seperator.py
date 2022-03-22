from typing import List
from src.models.note_info import NoteInfo
from src.models.range import Range
from src.models.ranges import ranges

from ordered_set import OrderedSet


def get_event_start_positions(note_infos: List[NoteInfo]):
    event_positions = list()

    for note_info in note_infos:
        event_positions.append(note_info.starting_beat)

    event_positions.sort()

    return OrderedSet(event_positions)

def get_notes_for_range(range: Range, remaining_notes: List[NoteInfo]):
    note_infos = list()
    remaining_notes = list(remaining_notes)

    notes_in_range = list(filter(lambda n : n.pitch >= range.min_note and n.pitch <= range.max_note, remaining_notes))
    event_positions = get_event_start_positions(notes_in_range)
    
    cur_pos = 0

    for position in event_positions:
        notes_at_position = list(filter(lambda n: n.starting_beat == position, notes_in_range))
        notes_at_position.sort(key=lambda n: n.pitch, reverse=True)

        highest_note = notes_at_position[0]
        if highest_note.starting_beat < cur_pos:
            continue

        note_infos.append(highest_note)
        remaining_notes.remove(highest_note)

        cur_pos = highest_note.starting_beat + highest_note.length


    return note_infos, remaining_notes

def get_tracks_by_range(note_infos: List[NoteInfo]):
    track_note_infos = list()
    remaining_notes = list(note_infos)

    for range_name in ranges:
        range = ranges[range_name]
        track_notes, remaining_notes = get_notes_for_range(range, remaining_notes)
        track_note_infos.append(track_notes)

    return track_note_infos