import enum

class Player(enum.Enum):
    black = 1
    white = 2

    # 이렇게 하면 그냥 attribute 콜하는 것 처럼 하면 getter 처리가 됨
    # a = Player(1); a.other
    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

from collections import namedtuple

class Point(namedtuple('Point', 'row col')): # == ['row', 'col'] == 'row, col'
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]
