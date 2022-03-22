from typing import List
from mido import Message, MidiFile, MidiTrack
from src.midi_message_generator import MidiMessageGenerator
from src.models.note_info import NoteInfo

def get_track(note_infos: List[NoteInfo], track_number):
    track = MidiTrack()
    track.append(Message('program_change', program=0, time=0))

    message_generator = MidiMessageGenerator(note_infos, track_number)
    for midi_message in message_generator.get_midi_note_messages():
        track.append(midi_message)

    return track

def get_midi_file(track_note_infos: List[List[NoteInfo]]):
    mid = MidiFile(type=1)

    for track_number, note_infos in enumerate(track_note_infos):
        track = get_track(note_infos, track_number)
        mid.tracks.append(track)
        
    return mid


def generate_midi(name, track_note_infos: List[List[NoteInfo]]):
    name = name.replace('.csv', '')

    midi_file = get_midi_file(track_note_infos)

    file_name = "midi/%s" % name
    midi_file.save(file_name)
