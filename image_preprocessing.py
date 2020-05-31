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
    validation_file = 'data/valid.p'
    testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    # image = X_train[342]
    # image = X_train[320]
    # image = X_train[543]
    # image = X_train[1543]
    # image = X_train[6522]
    # image = X_train[8522]
    # image = X_train[14832]
    # image = X_train[32000]
    # image = X_train[32834]
    # image = X_train[33834]
    # image = X_train[900]
    # image = X_train[1200]
    # image = X_train[2000]
    # image = X_train[2200]
    image = X_train[2400]
    print(y_train[2400])

    plt.imshow(image)
    plt.show()

    histo_gram = exposure.equalize_adapthist(image, clip_limit=0.1)

    plt.imshow(histo_gram)
    plt.show()

    color_image = histo_gram.astype('float32')

    p_image = np.uint8(color_image * 255)

    # plt.imshow(p_image)
    # plt.show()

    noised = cv2.fastNlMeansDenoisingColored(p_image, None, 20, 20, 3, 3)
    #
    plt.imshow(noised)
    plt.show()
