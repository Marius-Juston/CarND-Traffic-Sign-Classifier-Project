import os
import pickle
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import exposure
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2

tf.random.set_seed(42)


def preprocess(image, clip_limit=.1, h=20, h_color=20, template_window_size=3, search_window_size=3):
    image = exposure.equalize_adapthist(image, clip_limit=clip_limit).astype('float32')
    image = np.uint8(image * 255)
    image = cv2.fastNlMeansDenoisingColored(image, None, h, h_color, template_window_size, search_window_size)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return image


def preprocess_images(images):
    return np.array(list(preprocess(image) for image in images))


def save_images(x, y, output_file_name):
    np.savez(output_file_name, x=x, y=y)


def load_images(input_file_name):
    return np.load(input_file_name)


def create_model(image_shape, n_classes, dropout_out=.5, l2_rate=.001):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), input_shape=image_shape, activation='swish', padding='same',
                            kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(16, (3, 3), activation='swish', padding='same', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32, (3, 3), activation='swish', padding='same', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='swish', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_out))
    model.add(Dense(128, activation='swish', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_out))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def load_dataset():
    train_file = 'data/train.npz'
    test_file = 'data/test.npz'
    valid_file = 'data/valid.npz'

    if os.path.exists(train_file):
        dataset = load_images(train_file)
        X_train, y_train = dataset['x'], dataset['y']
    else:
        training_file = 'data/train.p'

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        X_train, y_train = train['features'], train['labels']

        X_train = preprocess_images(X_train)
        save_images(X_train, y_train, train_file.split('.')[0])

    if os.path.exists(test_file):
        dataset = load_images(test_file)
        X_test, y_test = dataset['x'], dataset['y']
    else:
        testing_file = 'data/test.p'

        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        X_test, y_test = test['features'], test['labels']

        X_test = preprocess_images(X_test)
        save_images(X_test, y_test, test_file.split('.')[0])

    if os.path.exists(valid_file):
        dataset = load_images(valid_file)
        X_valid, y_valid = dataset['x'], dataset['y']
    else:
        validation_file = 'data/valid.p'

        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        X_valid, y_valid = valid['features'], valid['labels']

        X_valid = preprocess_images(X_valid)
        save_images(X_valid, y_valid, valid_file.split('.')[0])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == '__main__':
    data_augmentor = ImageDataGenerator(rotation_range=15,
                                        zoom_range=.15,
                                        width_shift_range=.1,
                                        height_shift_range=.1,
                                        shear_range=.15,
                                        horizontal_flip=False,
                                        vertical_flip=False,
                                        fill_mode='nearest',
                                        # brightness_range=[.2, 1],
                                        )

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

    plt.imshow(X_valid[100])
    plt.show()
    plt.imshow(X_test[100])
    plt.show()
    plt.imshow(X_train[100])
    plt.show()

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(class_weights_dict)

    learning_rate = .01
    batch_size = 64
    epochs = 100
    decay = 0.0003

    image_shape, n_classes = X_train[0].shape, np.unique(y_train).shape[0]
    print(image_shape, n_classes)

    model = create_model(image_shape, n_classes)

    model.summary()

    optimizer = Adam(learning_rate=learning_rate, decay=decay)
    loss = SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, metrics=['accuracy'], loss=loss)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    H = model.fit(data_augmentor.flow(X_train, y_train, batch_size=batch_size),
                  # batch_size=batch_size,
                  steps_per_epoch=X_train.shape[0] // batch_size,
                  epochs=epochs,
                  class_weight=class_weights_dict,
                  validation_data=(X_valid, y_valid),
                  # use_multiprocessing=True,
                  callbacks=[early_stopping,
                             tensorboard_callback
                             ],
                  # verbose=2
                  )

    model.save('model' + datetime.now().strftime("%Y%m%d-%H%M%S"))
