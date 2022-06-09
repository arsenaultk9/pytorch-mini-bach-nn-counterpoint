class Range:
    def __init__(self, min_note: int, max_note: int):
        self.min_note = min_note
        self.max_note = max_note

    def range_and_silence_length(self):
        return (self.max_note - self.min_note + 1) + 1 # Range + 1 position for silence encoding


    def is_in_range(self, note_number):
        if note_number < self.min_note:
            return False

        if note_number > self.max_note:
            return False

        return True