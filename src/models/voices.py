from src.models.range import Range
from src.models.voice import Voice

voices = dict()

# Ranges calculated from min/max of stats.py with SAT 20 range and B 27 range.
voices['soprano'] = Voice('soprano', 3, Range(76, 96))
voices['alto'] = Voice('alto', 2, Range(64, 84))
voices['tenor'] = Voice('tenor', 1, Range(59, 79))
voices['bass'] = Voice('bass', 0, Range(43, 70))