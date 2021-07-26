import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(123)

X = np.load('../generated_games/features-40k.npy')
Y = np.load('../generated_games/labels-40k.npy')

samples = X.shape[0]
board_size = 9*9

X = X.reshape(samples, board_size)
Y = Y.reshape(samples, board_size)

train_samples = int(0.9 * samples)

X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(board_size,)))
model.add(Dense(500, activation='relu'))
model.add(Dense(board_size, activation='sigmoid'))
model.summary()

# model.compile(loss='mean_squared_error',
model.compile(loss='BinaryCrossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, batch_size=64, epochs=15, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
