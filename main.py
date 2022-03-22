import src.midi_generator as mg
import src.song_loader as sl
import src.song_parser as sp
import src.song_matrix_generator as smg
import src.song_range_seperator as srs

song = sl.load_song('data/5.kafka')
note_infos = sp.get_note_infos(song)

track_note_infos = srs.get_tracks_by_range(note_infos)

mg.generate_midi('file.mid', [note_infos] + track_note_infos)
