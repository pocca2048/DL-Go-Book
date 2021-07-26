from keras.layers.pooling import MaxPool2D
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout

np.random.seed(123)

X = np.load('../generated_games/features-40k.npy')
Y = np.load('../generated_games/labels-40k.npy')

samples = X.shape[0]
size = 9
input_shape = (size, size, 1)

X = X.reshape(samples, *input_shape)

train_samples = int(0.9 * samples)

X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Conv2D(filters=48, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape))
model.add(Dropout(0.25))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(size * size, activation='softmax'))
model.summary()

# model.compile(loss='mean_squared_error',
model.compile(loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, batch_size=64, epochs=100, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
