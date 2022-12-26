from typing import List
from src.models.note_info import NoteInfo
from src.data_augmentation.scale_data_augmenter import augment_scales

def augment_song_data(song_voices: List[List[NoteInfo]]):
    scale_augmentations = augment_scales(song_voices)
    return scale_augmentations