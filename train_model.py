from tensorflow.keras import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from read_smp import Samples
import os


def get_run_log_dir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


class ResidualUnit(Layer):

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(kwargs)
        self.activation = activations.get(activation)

        self.main_layers = [
            Conv2D(filters, 3, strides=strides, padding="same"),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same"),
            BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same"),
                BatchNormalization()
            ]

    def call(self, inputs, **kwargs):
        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)
        return self.activation(z + skip_z)


simple_model = tf.keras.models.Sequential([
        Conv2D(32, 3, activation='relu', padding="same", input_shape=(28, 28, 1)),
        MaxPooling2D(2),

        Conv2D(64, 3, activation='relu', padding="same"),
        Conv2D(64, 3, activation='relu', padding="same"),
        MaxPooling2D(2),

        Conv2D(128, 3, activation='relu', padding="same"),
        Conv2D(128, 3, activation='relu', padding="same"),
        MaxPooling2D(2),

        Conv2D(256, 3, activation='relu', padding="same"),
        Conv2D(256, 3, activation='relu', padding="same"),
        layers.MaxPooling2D(2),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(26, activation='softmax')
    ])

alex_net = models.Sequential([
    Conv2D(256, 5, activation='relu', padding="same", strides=1, input_shape=(28, 28, 1)),
    MaxPooling2D(3, strides=2, padding="valid"),
    Conv2D(384, 3, activation='relu', padding="same"),
    Conv2D(384, 3, activation='relu', padding="same"),
    Conv2D(256, 3, activation='relu', padding="same"),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

le_net = models.Sequential([
    Conv2D(6, 5, activation='tanh', strides=1, input_shape=(28, 28, 1)),
    AveragePooling2D(2, strides=2),
    Conv2D(16, 5, activation='tanh', strides=1),
    AveragePooling2D(2, strides=2),
    Conv2D(120, 4, activation='tanh', strides=1),
    Flatten(),
    Dense(84,activation='tanh'),
    Dense(26, activation='softmax')
])

res_net = models.Sequential()
res_net.add(Conv2D(32, 3, strides=2, input_shape=(28, 28, 1), padding="same"))
res_net.add(BatchNormalization())
res_net.add(Activation("relu"))
res_net.add(MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 32
for filters in [64] * 2 + [128] * 1:
    strides = 1 if filters == prev_filters else 2
    res_net.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
res_net.add(GlobalAvgPool2D())
res_net.add(Flatten())
res_net.add(Dense(128, activation='relu'))
res_net.add(Dropout(0.5))
res_net.add(Dense(64, activation='relu'))
res_net.add(Dropout(0.5))
res_net.add(Dense(26, activation="softmax"))

if __name__ == "__main__":
    root_log_dir = os.path.join(os.curdir, "my_logs")
    run_log_dir = get_run_log_dir()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_log_dir)

    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                  'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    smp_train = Samples()
    smp_train.loadFromSmpWxH('A-ZTrain.smp28')

    smp_test = Samples()
    smp_test.loadFromSmpWxH('A-ZTest.smp28')

    X_train, X_test, y_train, y_test = smp_train.imgs, smp_test.imgs, smp_train.labels, smp_test.labels

    print(le_net.summary())

    X_train = np.array(X_train).astype('float32')
    X_train /= 255

    X_test = np.array(X_test).astype('float32')
    X_test /= 255

    x_train = np.asarray(X_train)
    x_test = np.asarray(X_test)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)

    y_train = [dic[y] for y in y_train]
    y_test = [dic[y] for y in y_test]

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # le_net.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # le_net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=200, verbose=2, callbacks=[tensorboard_cb])
    #
    # le_net.save('A-9_model_le_net.h5')