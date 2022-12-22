from src.models.range import Range
from src.models.voice import Voice
import src.constants as constants

voices = dict()

if constants.APPLY_SCALE_AUGMENTATION:
    # Calculated ranges + scale augmentation buffer
    voices['soprano'] = Voice('soprano', 0, Range(61 - 5, 81 + 6))
    voices['alto'] = Voice('alto', 1, Range(56 - 5, 76 + 6))
    voices['tenor'] = Voice('tenor', 2, Range(51 - 5, 71 + 6))
    voices['bass'] = Voice('bass', 3, Range(36 - 5, 63 + 6))
else:
    # Ranges calculated from min/max of stats.py with SAT 20 range and B 27 range.
    voices['soprano'] = Voice('soprano', 0, Range(61, 81))
    voices['alto'] = Voice('alto', 1, Range(56, 76))
    voices['tenor'] = Voice('tenor', 2, Range(51, 71))
    voices['bass'] = Voice('bass', 3, Range(36, 63))