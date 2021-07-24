from dlgo.goboard import GameState
from dlgo.gotypes import Point

class Encoder:
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state: GameState):
        raise NotImplementedError()

    def encode_point(self, point: Point):
        raise NotImplementedError()

    def decode_point_index(self, index):
        raise NotImplementedError()

    def num_points(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()
        
import importlib

def get_encoder_by_name(name, board_size) -> Encoder:
    if isinstance(board_size, int):
        board_size = (board_size, board_size)
    module = importlib.import_module('dlgo.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)