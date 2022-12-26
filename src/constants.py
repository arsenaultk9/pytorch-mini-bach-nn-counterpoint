LOG_TENSORBOARD = False

SILENCE_INDEX = 0

SEGMENTS_PER_BEAT = 12
BEAT_SEGMENT_ACCEPTED_ROUNDING_ERROR = 1/50

MEASURES = 4
MEASURE_LENGTH = 16
SEQUENCE_LENGTH = MEASURES * MEASURE_LENGTH # Four measure of 16 notes segments

BATCH_LOG_INTERVAL = 50
VALID_PREDICTION_SAMPLE_RATE = 10 
OPTIMIZER_ADAM_LR = 1.0
EPOCHS = 180
BATCH_SIZE = 4

SHUFFLE_DATA = True
APPLY_LR_SCHEDULER = True

# Data augmentation
APPLY_SCALE_AUGMENTATION = True