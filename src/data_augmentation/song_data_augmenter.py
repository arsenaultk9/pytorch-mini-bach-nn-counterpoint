from typing import List
from src.models.note_info import NoteInfo
from src.data_augmentation.scale_data_augmenter import augment_scales
from src.data_augmentation.note_join_augmenter import augment_note_joins

def augment_song_data(song_voices: List[List[NoteInfo]]) -> List[List[List[NoteInfo]]]:
    scale_augmentations = augment_scales(song_voices)

    # Build on scale augmentation for note joins and note adds, but do not combine them.
    note_join_augmentations = []

    for scale_augmentation in scale_augmentations:
        note_join_augmentations += augment_note_joins(scale_augmentation)


    return note_join_augmentations