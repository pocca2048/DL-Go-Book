import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

import argparse

import h5py
from dlgo import agent, rl
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # 이걸 안하면 SGD가 안먹힘. keras가 tf에서인지 그냥에서인지에 따라 까다로움.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-in', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clipnorm', type=float, default=1.0)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--experience', nargs='+')

    args = parser.parse_args()
    learning_agent_filename = args.agent_in
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    clipnorm = args.clipnorm
    batch_size = args.bs

    learning_agent = agent.load_policy_agent(h5py.File(learning_agent_filename))
    for exp_filename in experience_files:
        exp_buffer = rl.load_experience(h5py.File(exp_filename))
        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            clipnorm=clipnorm,
            batch_size=batch_size)

    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
    # python ch10_train_pg.py --agent-in agent.hdf5 --agent-out new_agent.hdf5 --experience exp.hdf5
