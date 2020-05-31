import pickle
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def dataset_histogram():
    training_file = 'data/train.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    y_train = train['labels']

    counter = Counter(y_train)

    values = np.array(list(counter.items()))

    arr = values[values[:, 1].argsort()][::-1]

    plt.bar(arr[:, 0], arr[:, 1])
    plt.xlabel('Label')
    plt.ylabel("Frequency")
    plt.title("Sign Label based on Frequency")
    plt.savefig('project_report/histogram.png')

    plt.show()


def augmentation_viewer():
    training_file = 'data/train.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    image = train['features'][3432]

    cv2.imwrite('project_report/before_image.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    data_augmentor = ImageDataGenerator(rotation_range=15,
                                        zoom_range=.15,
                                        width_shift_range=.1,
                                        height_shift_range=.1,
                                        shear_range=.15,
                                        horizontal_flip=False,
                                        vertical_flip=False,
                                        fill_mode='nearest',
                                        brightness_range=[.2, 1],
                                        )

    data_augmentor.flow(image.reshape([1, 32, 32, 3]), batch_size=1, save_to_dir='project_report',
                        save_prefix="image", save_format="jpg")

    plt.show()


if __name__ == '__main__':
    dataset_histogram()
    augmentation_viewer()
