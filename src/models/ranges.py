from src.models.range import Range

ranges = dict()

# See https://github.com/omarperacha/TonicNet/blob/master/preprocessing/instruments.py for ranges

ranges['soprano'] = Range('soprano', 60, 81)
ranges['alto'] = Range('soprano', 53, 77)
ranges['tenor'] = Range('soprano', 45, 72)
ranges['bass'] = Range('soprano', 36, 64)