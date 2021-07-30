"""
각 조건을 만족하는 돌만 1인 plane으로 인코딩
1~3 plane: white whose 활로 == i
4~6 plane: black whose 활로 == i
7 plane: 패가 발생해서 움직이지 못하는 돌
"""
import numpy as np
from numpy import lib

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move, Point, GameState

class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size) -> None:
        self.board_width, self.board_height = board_size
        self.num_planes = 7

    def name(self):
        return 'sevenplane'

    def encode(self, game_state: GameState):
        board_tensor = np.zeros(self.shape())
        base_plane = {game_state.next_player: 0, game_state.next_player.other: 3}
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row+1, col+1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player, Move.play(p)):
                        board_tensor[6][row][col] = 1
                else:
                    liberty_plane = min(3, go_string.num_liberties) - 1
                    liberty_plane += base_plane[go_string.color]
                    board_tensor[liberty_plane][row][col] = 1
        return board_tensor

    def encode_point(self, point: Point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_height
        return Point(row+1, col+1)

    def num_points(self):
        return self.board_height * self.board_width

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

def create(board_size):
    return SevenPlaneEncoder(board_size)

