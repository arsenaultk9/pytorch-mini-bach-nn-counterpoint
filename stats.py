import pickle

with open('./data/JSB Chorales.pickle', 'rb') as file:
    dataset = pickle.load(file)

lowest_note = 127
highest_note = 0

for song in (dataset['train'] + dataset['test'] + dataset['valid']):
    for song_segment in song:
        if len(song_segment) == 0:
            continue

        highest_note = max(highest_note, max(song_segment))
        lowest_note = min(lowest_note, min(song_segment))


print(f'Lowest note: {lowest_note}')
print(f'Highest note: {highest_note}')