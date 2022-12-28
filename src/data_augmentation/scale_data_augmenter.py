from typing import List
from src.models.note_info import NoteInfo

def get_notes_transformed(song_voices: List[List[NoteInfo]], scale_transform: int) ->List[List[NoteInfo]]:
    voices_augmented = []

    for voice_notes in song_voices:
        notes_augmented = []

        for note_info in voice_notes:
            new_pitch = note_info.pitch + scale_transform
            note_augmented = note_info.with_new_pitch(new_pitch)

            notes_augmented.append(note_augmented)

        voices_augmented.append(notes_augmented)

    return voices_augmented

def augment_scales(song_voices: List[List[NoteInfo]]) -> List[List[List[NoteInfo]]]:
    scale_augmentations = []

    for scale_transform in range(-5, 7):
        scale_augmentation = get_notes_transformed(song_voices, scale_transform)
        scale_augmentations.append(scale_augmentation)

    return scale_augmentations