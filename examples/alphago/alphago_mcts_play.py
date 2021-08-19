import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '../..'))

import argparse
from collections import namedtuple

import h5py
from dlgo import scoring

from dlgo.agent import (AlphaGoMCTS, load_policy_agent, load_prediction_agent,
                        naive)
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.rl import load_value_agent

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution() # 이걸 안하면 SGD가 안먹힘. keras가 tf에서인지 그냥에서인지에 따라 까다로움.


BOARD_SIZE = 19
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board, BOARD_SIZE=BOARD_SIZE):
    for row in range(BOARD_SIZE, 0, -1):
        line = []
        for col in range(1, BOARD_SIZE + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:BOARD_SIZE])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        #if next_move.is_pass:
        #    print('%s passes' % name(game.next_player))
        game = game.apply_move(next_move)
        agents[game.next_player].reflect_move(next_move) # let opponent know that move occurred
        print_board(game.board)

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

def main():
    """
    alphago 설명
    구성요소 : 강하지만 느린 정책 신경망 + 약하지만 빠른 정책 신경망 + 가치 신경망
    * 빠른 정책 신경망 = 인간 기사의 대국 데이터를 사용해서 학습. test때 트리 탐색에서 rollout을 실행. 단말 노드를 평가하는데 사용.
    * 강한 정책 신경망 = 처음에는 인간 데이터로 학습하지만 이후 policy gradient 알고리즘을 사용한 자체 바둑을 두어 성능을 향상시킴.
    노드 선택 시 prior probability를 구하는데 사용.
    * 가치 신경망 = 자체 대국을 통해 생성된 경험 데이터를 사용해서 학습. 정책 롤아웃과 결합해서 단말 노드를 평가하는데 사용.

    알파고에서 수를 선택한다는 것 = 많은 시뮬레이션을 생성하고, 게임 트리를 돌아다니는 것. 시뮬레이션이 끝나면 가장 많이 방문된 노드를 선택.
    왜냐하면 시뮬레이션에서 Q(s,a) + U(s,a)를 더한 값이 최대인 move (Node)를 `선택`하기 때문.
    그러다가 단말 노드에 노달하면 강한 정책 신경망으로 prior를 생성한 후 이걸로 노드를 `확장`.
    가치 신경망과 빠른 정책 롤아웃의 결과를 합친 결합 가치 함수를 사용해서 단말 노드를 평가.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=1)
    args = parser.parse_args()

    fast_policy = load_prediction_agent(h5py.File('alphago_sl_policy.hdf5', 'r'))
    strong_policy = load_policy_agent(h5py.File('alphago_rl_policy.hdf5', 'r'))
    value = load_value_agent(h5py.File('alphago_value.hdf5', 'r'))

    agent1 = AlphaGoMCTS(strong_policy, fast_policy, value, num_simulations=1, depth=5, rollout_limit=5)
    # agent2 = AlphaGoMCTS(strong_policy, fast_policy, value)
    agent2 = naive.RandomBot()


    wins = 0
    losses = 0
    color1 = Player.black
    for i in range(args.num_games):
        print('Simulating game %d/%d...' % (i + 1, args.num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            wins += 1
        else:
            losses += 1
        color1 = color1.other
    print('Agent 1 record: %d/%d' % (wins, wins + losses))

    # TODO: register in frontend

if __name__ == "__main__":
    main()
