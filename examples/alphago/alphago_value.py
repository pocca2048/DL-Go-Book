import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '../..'))

from dlgo.networks.alphago import alphago_model
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl import ValueAgent, load_experience
import h5py
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # 이걸 안하면 SGD가 안먹힘. keras가 tf에서인지 그냥에서인지에 따라 까다로움.

def main():
    rows, cols = 19, 19
    encoder = AlphaGoEncoder()
    input_shape = (encoder.num_planes, rows, cols)
    alphago_value_network = alphago_model(input_shape)

    alphago_value = ValueAgent(alphago_value_network, encoder)
    # end::init_value[]

    # tag::train_value[]
    experience = load_experience(h5py.File('alphago_rl_experience.hdf5', 'r'))

    alphago_value.train(experience)

    with h5py.File('alphago_value.hdf5', 'w') as value_agent_out:
        alphago_value.serialize(value_agent_out)
    # end::train_value[]

if __name__ == "__main__":
    main()
    # after alphago_policy_rl.py
    # python alphago_value.py