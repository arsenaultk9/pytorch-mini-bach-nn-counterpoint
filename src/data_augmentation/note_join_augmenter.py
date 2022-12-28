from typing import List
import random

from src.models.note_info import NoteInfo

def get_notes_joined(song_voices: List[List[NoteInfo]], modification_prob: float) ->List[List[NoteInfo]]:
    voices_notes_joined = []

    for voice_notes in song_voices:
        voice_notes_joined = []

        for note in voice_notes:
            # Only consider notes not on measure beats that are smaller than one beat
            if note.starting_beat % 1.0 == 0 or note.length >= 1.0:
                voice_notes_joined.append(note.copy()) # Copy notes to avoid mutation in other augmentations.
                continue

            keep_prob = random.uniform(0, 1)
            keep = keep_prob > modification_prob

            if not keep:
                # Join note duration for last note.
                last_note = voice_notes_joined[-1]
                last_note.length += note.length
                continue

            voice_notes_joined.append(note.copy())


        voices_notes_joined.append(voice_notes_joined)

    return voices_notes_joined

def augment_note_joins(song_voices: List[List[NoteInfo]]) -> List[List[List[NoteInfo]]]:
    # Keep vanilla data
    note_join_augmentations = [song_voices]

    for _ in range(3):
        modification_prob = random.uniform(0, 1)
        note_join_augmentations.append(get_notes_joined(song_voices, modification_prob))

    return note_join_augmentations