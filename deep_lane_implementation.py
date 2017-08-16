from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout
import numpy
import os


def train_model(X, Y, epochs=60, batch_size=128, fname="trained_data.h5"):
    model = Sequential()
    model.add(Conv2D(32, 18, (6, 6), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(64, 10, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(317, activation="softmax"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    model.save(fname)
    return model


numpy.random.seed(7)

epochs = 60
X = numpy.array()  # TODO: Load image data
Y = numpy.array()

if os.path.exists("trained_data.h5"):
    model = load_model("trained_data.h5")
else:
    model = train_model(X, Y, epochs)
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

test_x = ""  # TODO: Load an image
predictions = model.predict(test_x)
