import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '../..'))

import h5py
import tensorflow as tf

from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl.simulate import experience_simulation

tf.compat.v1.disable_eager_execution() # 이걸 안하면 SGD가 안먹힘. keras가 tf에서인지 그냥에서인지에 따라 까다로움.

def main():
    encoder = AlphaGoEncoder()

    sl_agent = load_prediction_agent(h5py.File('alphago_sl_policy.hdf5'))
    sl_opponent = load_prediction_agent(h5py.File('alphago_sl_policy.hdf5'))

    alphago_rl_agent = PolicyAgent(sl_agent.model, encoder) # 유의) Policy Agent를 사용한다.
    opponent = PolicyAgent(sl_opponent.model, encoder)
    

    # tag::run_simulation[]
    num_games = 5 # 1000
    experience = experience_simulation(num_games, alphago_rl_agent, opponent)

    alphago_rl_agent.train(experience)

    with h5py.File('alphago_rl_policy.hdf5', 'w') as rl_agent_out:
        alphago_rl_agent.serialize(rl_agent_out)

    with h5py.File('alphago_rl_experience.hdf5', 'w') as exp_out:
        experience.serialize(exp_out)
    # end::run_simulation[]

if __name__ == "__main__":
    main()
    # after alphago_policy_sl.py
    # python alphago_policy_rl.py
