from src.models.note_info import NoteInfo


def test_is_on_at_beat_when_before_beat_is_false():
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(0)

    assert is_on_beat is False


def test_is_on_at_beat_when_after_beat_is_false(): 
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(2)

    assert is_on_beat is False


def test_is_on_at_beat_when_on_start_beat_is_true():
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(1)

    assert is_on_beat is True


def test_is_on_at_beat_when_very_slightly_before_beat_is_true():
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(0.999)

    assert is_on_beat is True


def test_is_on_at_beat_when_on_last_before_twelth_of_beat_is_true():
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(1 + (11/12))

    assert is_on_beat is True


def test_is_on_at_beat_when_on_last_before_twelth_of_beat_and_little_over_is_true():
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(1 + (11/12) + 0.01)

    assert is_on_beat is True


def test_is_on_at_beat_when_end_of_beat_is_false():
    note = NoteInfo.create(1)
    is_on_beat = note.is_on_at_beat(2)

    assert is_on_beat is False
