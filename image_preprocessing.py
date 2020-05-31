import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


def preprocess(image, clip_limit=.1, h=20, h_color=20, template_window_size=3, search_window_size=3):
    image = exposure.equalize_adapthist(image, clip_limit=clip_limit).astype('float32')
    image = np.uint8(image * 255)
    image = cv2.fastNlMeansDenoisingColored(image, None, h, h_color, template_window_size, search_window_size)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return image


def preprocess_images(images):
    # return np.fromiter((preprocess(image) for image in images), dtype=np.float32)
    return np.array(list(preprocess(image) for image in images))


if __name__ == '__main__':
    training_file = 'data/train.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    index = 30552

    image = X_train[index]
    print(y_train[index])

    plt.imshow(image)
    plt.show()

    image = preprocess(image)

    plt.imshow(image)
    plt.show()

    images = preprocess_images(X_train)

    plt.imshow(image[index])
    plt.show()

    print(images.shape)
    print(images.dtype)
