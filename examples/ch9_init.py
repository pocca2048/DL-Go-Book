import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

import argparse

import h5py
from dlgo import agent, encoders, networks
from keras.layers import Activation, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', type=int, default=19)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    board_size = args.board_size
    output_file = args.output_file

    encoder = encoders.simple.SimpleEncoder((board_size, board_size))
    model = Sequential()
    for layer in networks.small.layers(encoder.shape()):
        model.add(layer)
    model.add(Dense(encoder.num_points()))
    model.add(Activation('softmax'))
    new_agent = agent.PolicyAgent(model, encoder)

    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()
