# This scripts demonstrates all the steps to create and train an
# AGZ-style bot.
# For practical purposes, you would separate this script into multiple
# parts (for initializing, generating self-play games, and training).
# You'll also need to run for many more rounds.
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '../..'))

import argparse

import h5py
import tensorflow as tf
from dlgo import kerasutil, scoring, zero
from dlgo.goboard_fast import GameState, Player

from model import init_model
from examples.alphago.alphago_mcts_play import print_board

tf.compat.v1.disable_eager_execution() # 이걸 안하면 SGD가 안먹힘. keras가 tf에서인지 그냥에서인지에 따라 까다로움.

def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)
    print_board(game.board, BOARD_SIZE=board_size)

    # Give the reward to the right agent.
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=5)
    parser.add_argument('--board-size', '-bsz', type=int, default=9)
    args = parser.parse_args()

    # Initialize a zero agent
    board_size = args.board_size
    encoder = zero.ZeroEncoder(board_size)

    model = init_model(encoder=encoder)

    # Create two agents from the model and encoder.
    # 10 is a very small value for rounds_per_move. To train a strong
    # bot, you should run at least a few hundred rounds per move.
    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    # In real training, you should simulate thousands of games for each
    # training batch.
    for i in range(args.num_games):
        simulate_game(board_size, black_agent, c1, white_agent, c2)

    exp = zero.combine_experience([c1, c2])
    print('Simulation Finished. Starting Training!')
    black_agent.train(exp, 0.01, 2048)

    with h5py.File('alphazero.hdf5', 'w') as agent_out:
        agent_out.create_group('model')
        kerasutil.save_model_to_hdf5_group(black_agent.model, agent_out['model'])


if __name__ == '__main__':
    main()
