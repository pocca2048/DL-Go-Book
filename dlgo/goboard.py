# https://stackoverflow.com/a/33533514 for GoString hint
from __future__ import annotations

import copy
from typing import List

from dlgo import zobrist
from dlgo.gotypes import Player, Point  # PYTHON IMPORTS SUCKS
from dlgo.scoring import compute_game_result


class Move():
    def __init__(self, point: Point = None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    # c++의 function overloading 대신에 다른 constructor를 위한 파이썬의 방식
    @classmethod
    def play(cls, point: Point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)

class GoString():
    """
    이음 = 같은 색 돌의 연결된 그룹.
    돌 하나마다 보면 오래걸리므로 묶어서 보자.
    """
    def __init__(self, color: Player, stones: List[Point], liberties: List[Point]):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties) # 활로

    def without_liberty(self, point: Point):
        """
        remove_liberty를 대체한다.
        zobrist 해싱을 적용함에 따라 mutable에서 immutable로 바뀌므로 liberty가 없는 새로운 string을 리턴.
        """
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point: Point):
        """
        add_liberty를 대체한다.
        zobrist 해싱을 적용함에 따라 mutable에서 immutable로 바뀌므로 liberty가 없는 새로운 string을 리턴.
        """
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, go_string: GoString):
        """돌을 놓아 2개의 그룹을 연결한 경우 호출. 이해안됨"""
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones
        )

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties
    
class Board():
    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {} # 놓인 돌 (Point -> gostring) 들을 저장
        self._hash = zobrist.EMPTY_BOARD

    def is_on_grid(self, point: Point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point: Point):
        """
        돌이 있으면 흑/백 Player 반환. 아니면 None
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point: Point) -> GoString:
        """
        돌이 있으면 GoString 반환.
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    
    def _replace_string(self, new_string: GoString):
        """바둑판을 갱신함."""
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string: GoString):
        """
        하나의 string (이음)을 제거하면 주변에 있던 애들은 활로 (liberty)가 늘어난다.
        """
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))
            
            self._grid[point] = None
            # 돌을 제거하는 것 = 돌의 해시값을 비적용하는 것.
            self._hash ^= zobrist.HASH_CODE[point, string.color]

    def zobrist_hash(self):
        return self._hash
        
    def place_stone(self, player: Player, point: Point) -> None:
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []

        # 지금 놓을 stone을 이음화 -> new_string
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor) # 주변이 비었으면 liberty
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string) # 주변에 같은색 이음이 있으면 adjacent_same_color에 추가
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string) # 주변에 다른색 이음이 있으면 adjacent_opposite_color에 추가
        new_string = GoString(player, [point], liberties)

        # 같은 색의 근접한 이음을 합친다.
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        # grid 갱신.
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        self._hash ^= zobrist.HASH_CODE[point, player]

        for other_color_string in adjacent_opposite_color:
            replacement = other_color_string.without_liberty(point)
            # 다른 색의 이음에서 활로를 제거한다.
            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            # 다른 색의 이음의 활로가 0이 되면 이를 제거한다.
            else:
                self._remove_string(other_color_string)
    

# 3.2 대국 현황 기록과 반칙수 확인

class GameState():
    def __init__(self, board: Board, next_player: Player, previous: GameState, move: Move) -> None:
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())}
            )

        self.last_move = move

    def apply_move(self, move: Move):
        """하기 전에 is_valid_move 호출해야 함."""
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        """is_resign이거나 2번 연속 is_pass이면 over"""
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def is_move_self_capture(self, player, move):
        """자충수 규칙"""
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point) # 먼저 둬서 잡는 수인지 확인
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0 # 나중에 활로 수 확인.

    @property
    def situation(self):
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player: Player, move: Move):
        """패 규칙을 위반하는지 확인. 과거의 모든 state를 확인하므로 매우 느렸었지만 hash로 개선.
        이것보다 빠르게 하고 싶으면 github의 goboard_fast.py를 확인. 가독성은 떨어지지만 빠름."""
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_states

    def is_valid_move(self, move: Move):
        """
        1. 거기가 비었는지 확인
        2. 자충수가 아닌지 확인 (자충수 self-capture = 활로가 1개이면서 잡는 수가 아닐때 거기에 두는것)
        3. 바둑 규칙에 위배되는지 확인
        """
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
            self.board.get(move.point) is None and
            not self.is_move_self_capture(self.next_player, move) and
            not self.does_move_violate_ko(self.next_player, move)
        )
    
    # copied from github. not in the book.
    def legal_moves(self) -> List[Move]:
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        # These two moves are always legal.
        moves.append(Move.pass_turn())
        moves.append(Move.resign())

        return moves

    # copied from github. not in the book.
    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self)
        return game_result.winner

# 3.3 게임 종료
# 집 = 모든 직선 면의 점과 최소 네 대각선 중 3개 이상의 접한 점이 본인 색의 돌로 채워진 빈 점.
# 모서리에 만들어지는 집의 경우는 따로 처리해야 함. 모든 대각선에 인접한 점에는 같은 돌이 있어야 함.
# 일부 유효한 집의 정의가 빠진 정의.

