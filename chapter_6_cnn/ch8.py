import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import small
from pathlib import Path
# end::e2e_imports[]

def main():
    # tag::e2e_processor[]
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    X, y = processor.load_go_data(num_samples=100)
    # end::e2e_processor[]

    # tag::e2e_model[]
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = small.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(X, y, batch_size=128, epochs=20, verbose=1)
    # end::e2e_model[]

    # tag::e2e_agent[]
    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(Path(".").absolute().parents[0] / "agents/deep_bot.h5", "w")
    deep_learning_bot.serialize(model_file)
    # end::e2e_agent[]

    # tag::e2e_load_agent[]
    model_file = h5py.File(Path(".").absolute().parents[0] / "agents/deep_bot.h5", "r")
    # model_file = h5py.File(Path("../agents/deep_bot.h5"), "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()

def run_server():
    model_file = h5py.File(Path(".").absolute().parents[0] / "agents/deep_bot.h5", "r")
    # model_file = h5py.File(Path("../agents/deep_bot.h5"), "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()

if __name__ == "__main__":
    # main()
    run_server() # http://127.0.0.1:5000/static/play_predict_19.html
