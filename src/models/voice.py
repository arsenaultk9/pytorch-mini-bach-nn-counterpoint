from src.models.range import Range

class Voice:
    def __init__(self, name: str, tuple_index: int, range: Range):
        self.name = name
        self.tuple_index = tuple_index
        self.range = range


    # TODO: Manage better logic for note in range. Not as simple as getting note at index.
    def get_note_in_segment(self, song_segment):
        if len(song_segment) - 1 < self.tuple_index:
            return - 1

        return song_segment[self.tuple_index]