from dlgo.gotypes import Point

# 3.3 게임 종료
# 집 = 모든 직선 면의 점과 최소 네 대각선 중 3개 이상의 접한 점이 본인 색의 돌로 채워진 빈 점.
# 모서리에 만들어지는 집의 경우는 따로 처리해야 함. 모든 대각선에 인접한 점에는 같은 돌이 있어야 함.
# 일부 유효한 집의 정의가 빠진 정의.

def is_point_an_eye(board, point, color):
    """
    집 = 모든 직선 면의 점과 최소 네 대각선 중 3개 이상의 접한 점이 본인 색의 돌로 채워진 빈 점.
    모서리에 만들어지는 집의 경우는 따로 처리해야 함. 모든 대각선에 인접한 점에는 같은 돌이 있어야 함.
    일부 유효한 집의 정의가 빠진 정의.
    *******
    *-*---*
    *******
    이런거 안될듯.
    """
    if board.get(point) is not None:
        return False
    # 모든 근접한 점에 자신의 돌이 놓여 있음.
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
            
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    return friendly_corners >= 3