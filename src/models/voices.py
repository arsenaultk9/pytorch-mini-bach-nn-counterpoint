from src.models.range import Range
from src.models.voice import Voice

voices = dict()

# Ranges calculated from min/max of stats.py with SAT 20 range and B 27 range.
voices['soprano'] = Voice('soprano', 0, Range(61, 81))
voices['alto'] = Voice('alto', 1, Range(56, 76))
voices['tenor'] = Voice('tenor', 2, Range(51, 71))
voices['bass'] = Voice('bass', 3, Range(36, 63))