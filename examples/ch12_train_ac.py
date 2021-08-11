import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

import argparse

import h5py

from dlgo import rl
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-in', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    learning_agent_filename = args.agent_in
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    batch_size = args.bs

    learning_agent = rl.load_ac_agent(h5py.File(learning_agent_filename))
    for exp_filename in experience_files:
        exp_buffer = rl.load_experience(h5py.File(exp_filename))
        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)

    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
    # after ch12_self_play.py
    # python ch12_train_ac.py --agent-in ac_v1.hdf5 --agent-out ac_v2.hdf5 ac_exp_001.hdf5
    # after this, evaluate whether ac_v1 is better or ac_v2 is better and iterate process.
