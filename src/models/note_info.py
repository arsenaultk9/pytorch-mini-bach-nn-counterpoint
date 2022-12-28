import src.constants as constants

# Data columns:
# Starting beat(Quarter notes),
# Pitch (0, 127),
# relative Pitch,
# Length(Quarter notes),
# Midi Channel


class NoteInfo:
    def __init__(self, note_data):
        self.starting_beat = float(note_data[0])
        self.pitch = int(note_data[1])
        self.length = float(note_data[2])
        self.midi_channel = note_data[3]

    def is_on_at_beat(self, beat):
        if (beat < self.starting_beat - constants.BEAT_SEGMENT_ACCEPTED_ROUNDING_ERROR):
            return False

        inclusive_end = self.starting_beat + \
            self.length - (1/constants.SEGMENTS_PER_BEAT) + \
            constants.BEAT_SEGMENT_ACCEPTED_ROUNDING_ERROR

        if (beat > inclusive_end):
            return False

        return True

    def with_new_pitch(self, new_pitch):
        return NoteInfo.create(self.starting_beat, self.length, new_pitch)

    def copy(self):
        return NoteInfo.create(self.starting_beat, self.length, self.pitch)


    def __str__(self):
        return "starting_beat: {}, length: {}, pitch: {}".format(self.starting_beat, self.length, self.pitch)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def create(cls, starting_beat, length=1, pitch=0):
        return cls([
            starting_beat,
            pitch,
            length,
            0
        ])
