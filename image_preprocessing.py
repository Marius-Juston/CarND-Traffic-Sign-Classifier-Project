import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

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
