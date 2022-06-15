from typing import Tuple
from src.models.voice import Voice


class SongNoteRangeTracker:
    def __init__(self, voice: Voice):
        self.voice = voice
        self.prev_note = None
        self.prev_song_segment = None

    def apply_get_note(self, note, song_segment):
        self.prev_note = note
        self.prev_song_segment = song_segment

        return int(note)

    # Note: Could use a better technique, but since we only do a forward pass, this is the least problematic resolution.
    def get_first_note_missing_in_tuple(self, song_segment):
        notes_in_segment = len(song_segment)
        tuple_index_delta = 4 - notes_in_segment
        new_tuple_index = self.voice.tuple_index - tuple_index_delta

        if new_tuple_index < 0:
            return -1

        return song_segment[new_tuple_index]


    def is_note_closest_of_previous_notes(self, distance_and_prev_note):
        target_note = distance_and_prev_note[1]
        min_distance = distance_and_prev_note[0]

        for prev_note in self.prev_song_segment:
            distance = abs(prev_note - target_note)
            min_distance = min(min_distance, distance)

        return distance_and_prev_note[0] <= min_distance

    def get_most_probable_note_in_segment(self, song_segment):
        if len(song_segment) == 0:
            return -1

        if len(song_segment) == 4:
            return song_segment[self.voice.tuple_index]

        if self.prev_note in song_segment:
            return self.prev_note

        if self.prev_song_segment is None:
            return self.get_first_note_missing_in_tuple(song_segment)

        closest_to_prev_note = list(map(lambda n : (abs(self.prev_note - n), n), song_segment))
        closest_to_prev_note.sort(key=lambda dn : dn[0])
        distance_and_target_note = closest_to_prev_note[0]

        if not self.is_note_closest_of_previous_notes(distance_and_target_note):
            return -1

        return distance_and_target_note[1]

    def get_next_note(self, song_segment: Tuple):
        note_candidate = self.get_most_probable_note_in_segment(song_segment)

        if not self.voice.range.is_in_range(note_candidate):
            return self.apply_get_note(-1, song_segment)

        return self.apply_get_note(note_candidate, song_segment)
