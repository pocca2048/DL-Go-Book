from typing import Tuple
import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import GameState, Point

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size) -> None:
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def name(self):
        return 'oneplane'

    def encode(self, game_state: GameState) -> np.array:
        '''
        현재 player의 돌이 놓여있다면 +1, 현재 player의 상대 player의 돌이 놓여있다면 -1.
        빈 점에는 0
        '''
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(r+1, c+1)
                go_string = game_state.board.get_go_string(point=p)
                if go_string is None:
                    continue
                if go_string.color == next_player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1
        return board_matrix
        
    def encode_point(self, point: Point) -> int:
        '''
        바둑판의 점 위치를 정수형 인덱스로 변환한다.
        '''
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index: int) -> Point:
        '''
        정수형 인덱스를 바둑판의 점 위치로 변환한다.
        '''
        row = index // self.board_width
        col = index % self.board_width
        return Point(row + 1, col + 1)

    def num_points(self) -> int:
        return self.board_height * self.board_width

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

def create(board_size):
    return OnePlaneEncoder(board_size)