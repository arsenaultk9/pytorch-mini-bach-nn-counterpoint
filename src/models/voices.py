from src.models.range import Range
from src.models.voice import Voice

voices = dict()

# TODO: Adjust ranges from dataset

voices['soprano'] = Voice('soprano', 3, Range(60, 94))
voices['alto'] = Voice('alto', 2, Range(53, 77))
voices['tenor'] = Voice('tenor', 1, Range(45, 72))
voices['bass'] = Voice('bass', 0, Range(36, 64))