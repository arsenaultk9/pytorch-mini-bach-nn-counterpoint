import pickle
from src.models.song_note_range_tracker import SongNoteRangeTracker
from src.models.voices import voices


with open('./data/JSB Chorales.pickle', 'rb') as file:
    dataset = pickle.load(file)

lowest_note = 127
highest_note = 0

voice_ranges = {
    'soprano': {'lowest_note': 127, 'highest_note': 0},
    'alto': {'lowest_note': 127, 'highest_note': 0},
    'tenor': {'lowest_note': 127, 'highest_note': 0},
    'bass': {'lowest_note': 127, 'highest_note': 0}
}

for song in (dataset['train'] + dataset['test'] + dataset['valid']):
    for song_segment in song:
        if len(song_segment) == 0:
            continue

        highest_note = max(highest_note, max(song_segment))
        lowest_note = min(lowest_note, min(song_segment))

    for voice_name in voice_ranges.keys():
        voice = voices[voice_name]
        note_range_tracker = SongNoteRangeTracker(voice)

        for song_segment in song:
            voice_note = note_range_tracker.get_next_note(song_segment)

            if voice_note == -1:
                continue

            voice_lowest_note = voice_ranges[voice_name]['lowest_note']
            voice_ranges[voice_name]['lowest_note'] = min(voice_lowest_note, voice_note)

            voice_highest_note = voice_ranges[voice_name]['highest_note']
            voice_ranges[voice_name]['highest_note'] = max(voice_highest_note, voice_note)
        




print(f'Global lowest note: {lowest_note}')
print(f'Global highest note: {highest_note}')


for voice_name, range in voice_ranges.items():
    print(f"{voice_name} lowest note: {range['lowest_note']}")
    print(f"{voice_name} highest note: {range['highest_note']}")
