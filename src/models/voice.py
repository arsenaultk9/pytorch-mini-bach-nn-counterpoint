from src.models.range import Range

class Voice:
    def __init__(self, name: str, tuple_index: int, range: Range):
        self.name = name
        self.tuple_index = tuple_index
        self.range = range


    def get_note_from_tensor_position(self, tensor_position: int):
        if tensor_position == 0:
            return None

        return tensor_position + self.range.min_note - 1