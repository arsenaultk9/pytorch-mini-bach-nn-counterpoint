from typing import List
from src.models.note_info import NoteInfo
from mido import Message
from ordered_set import OrderedSet

# The default tempo is 120 BPM.
# (500000 microseconds per beat (quarter note).)
DEFAULT_TEMPO = 500000
PULSE_PER_QUARTE_NOTE = 480

beat_length_ms = PULSE_PER_QUARTE_NOTE  # NOTE: This is good.


def get_event_positions(note_infos: List[NoteInfo]):
    event_positions = list()

    for note_info in note_infos:
        event_positions.append(note_info.starting_beat)
        event_positions.append(note_info.starting_beat + note_info.length)

    event_positions.sort()

    return OrderedSet(event_positions)

def get_position_delta_from_previous_notes(position: float, previous_notes: List[NoteInfo]):
    if len(previous_notes) == 0:
        return position

    smallest_distance_from_position = 999999 # Big enough value so that no distance between two notes equal this.
    for note in previous_notes:
        note_start = note.starting_beat
        if note_start > position:
            continue

        smallest_distance_from_position = min(smallest_distance_from_position, position - note_start)

        note_end = note.starting_beat + note.length
        if note_end >= position:
            continue

        smallest_distance_from_position = min(smallest_distance_from_position, position - note_end)


    return smallest_distance_from_position

# Get notes at position in a forward pass of all notes till note start and end after position
def get_notes_for_position(position: float, note_infos: List[NoteInfo]):
    for note in note_infos:
        note_start = note.starting_beat
        note_end = note.starting_beat + note.length

        # After position, continue in next iteration
        if note_end > position and note_start > position:
            break

        if note_start == position:
            yield note

        if note_end == position:
            yield note


def get_notes_and_midi_events_for_position(position: float,
                                           previous_notes: List[NoteInfo],
                                           note_infos: List[NoteInfo],
                                           track_number):
    exhausted_delta = False
    position_delta = get_position_delta_from_previous_notes(position, previous_notes)

    used_notes = []
    midi_events = []

    for note in get_notes_for_position(position, note_infos):
        used_notes.append(note)

        note_start = note.starting_beat

        relative_position = int(position_delta * beat_length_ms) if not exhausted_delta else 0
        message_type = 'note_on' if note_start == position else 'note_off'
        note_pitch = note.pitch
        velocity = 64 if note_start == position else 0

        exhausted_delta = True

        message = Message(message_type, note=note_pitch, velocity=velocity, time=relative_position, channel=track_number)
        midi_events.append(message)



    return used_notes, midi_events



class MidiMessageGenerator():
    def __init__(self, note_infos: List[NoteInfo], track_number):
        self.note_infos = note_infos
        self.track_number = track_number

    def get_midi_note_messages(self):

        event_positions = get_event_positions(self.note_infos)
        previous_notes = []

        for cur_pos in event_positions:
            used_notes, midi_events = get_notes_and_midi_events_for_position(cur_pos, previous_notes, self.note_infos,
            self.track_number)

            previous_notes = used_notes
            for midi_event in midi_events:
                yield midi_event
