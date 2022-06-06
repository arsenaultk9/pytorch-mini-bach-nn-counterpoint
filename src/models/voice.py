from src.models.range import Range

class Voice:
    def __init__(self, name: str, tuple_index: int, range: Range):
        self.name = name
        self.tuple_index = tuple_index
        self.range = range