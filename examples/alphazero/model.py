from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input
from keras.models import Model


def init_model(encoder):
    board_input = Input(shape=encoder.shape(), name='board_input')

    pb = board_input

    # 4 conv layers with batch normalization
    for i in range(4):
        pb = Conv2D(64, (3, 3),
            padding='same',
            data_format='channels_first')(pb)
        pb = BatchNormalization(axis=1)(pb)
        pb = Activation('relu')(pb)

    # Policy output
    policy_conv = Conv2D(2, (1, 1), data_format='channels_first')(pb)
    policy_batch = BatchNormalization(axis=1)(policy_conv)
    policy_relu = Activation('relu')(policy_batch)
    policy_flat = Flatten()(policy_relu)
    policy_output = Dense(encoder.num_moves(), activation='softmax')(
        policy_flat)

    # Value output
    value_conv = Conv2D(1, (1, 1), data_format='channels_first')(pb)
    value_batch = BatchNormalization(axis=1)(value_conv)
    value_relu = Activation('relu')(value_batch)
    value_flat = Flatten()(value_relu)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(
        inputs=[board_input],
        outputs=[policy_output, value_output])

    return model