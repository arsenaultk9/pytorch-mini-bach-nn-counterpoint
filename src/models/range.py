class Range:
    def __init__(self, min_note: int, max_note: int):
        self.min_note = min_note
        self.max_note = max_note

    def range_and_silence_length(self):
        return self.max_note - self.min_note + 1