from src.models.note_info import NoteInfo


def get_measure_length(measure):
    signature = measure['MeasureSignature']
    beats_per_measure = signature['BeatsPerMeasure']
    note_value_beat = signature['NoteValueOfBeat']
    measure_length = (beats_per_measure * 4) / note_value_beat

    return measure_length

def get_rational_to_float(rational):
    if('/' not in rational):
        return float(rational)

    numerators = rational.split('/')

    return float(numerators[0]) / float(numerators[1])

def get_note_info_from_unit(unit, current_measure_pos):
    starting_beat = current_measure_pos + get_rational_to_float(unit['PositionInMeasure'])
    length = get_rational_to_float(unit['Lenght'])
    pitch = unit['Note']['NoteFromMidiNumber']

    return NoteInfo.create(starting_beat, length, pitch)


def get_measure_note_infos(measure, current_measure_pos):
    note_infos = []
    
    for unit in measure['Units']:
        if 'MeasureSilence' in unit['$type']:
            continue
            
        note_info = get_note_info_from_unit(unit, current_measure_pos)
        note_infos.append(note_info)

    return note_infos

def get_note_infos(song):
    note_infos = []

    for track in song['InstrumentTracks']:
        current_measure_pos = 0

        for measure in track['Measures']:
            measure_length = get_measure_length(measure)
            measure_note_infos = get_measure_note_infos(measure, current_measure_pos)

            note_infos.extend(measure_note_infos)

            current_measure_pos += measure_length

    return note_infos
