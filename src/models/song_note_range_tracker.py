from typing import Tuple
from src.models.voice import Voice


class SongNoteRangeTracker:
    def __init__(self, voice: Voice):
        self.voice = voice
        self.prev_note = None
        self.prev_song_segment = None

    # TODO: Rendu ici <------------------------------------------------------------
    # Make this a track with previous note instead of this simple logic.
    def get_next_note(self, song_segment: Tuple):
        if len(song_segment) - 1 < self.voice.tuple_index:
            return - 1

        return song_segment[self.voice.tuple_index]
