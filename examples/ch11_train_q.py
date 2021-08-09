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
    """
    ch 11 q-learning
    Q(s, a) = 경기 결과 (이긴 경우 1, 진 경우 -1)
    이전에는 실제 대국에서의 수를 예측하거나, 이기는 player의 수를 예측했지만
    지금은 대국에서의 상태와 수가 주어졌을 때 이길지 질지를 예측한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-in', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--experience', nargs='+')

    args = parser.parse_args()
    learning_agent_filename = args.agent_in
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    batch_size = args.bs

    learning_agent = rl.load_q_agent(h5py.File(learning_agent_filename))
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
    # after ch11_self_play.py
    # python ch11_train_q.py --agent-in qmodel.hdf5 --agent-out new_qmodel.hdf5 --experience q_exp.hdf5
