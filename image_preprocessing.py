import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


def preprocess(image, clip_limit=.1, h=20, h_color=20, template_window_size=3, search_window_size=3):
    # cv2.imwrite('project_report/preprocessing_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = exposure.equalize_adapthist(image, clip_limit=clip_limit).astype('float32')
    image = np.uint8(image * 255)
    # cv2.imwrite('project_report/preprocessing_histogram_equalized_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = cv2.fastNlMeansDenoisingColored(image, None, h, h_color, template_window_size, search_window_size)
    # cv2.imwrite('project_report/preprocessing_denoised_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return image


def preprocess_images(images):
    return np.array(list(preprocess(image) for image in images))


def save_images(images_numpy, output_file_name):
    np.save(output_file_name, images_numpy)


def load_images(input_file_name):
    return np.load(input_file_name)


if __name__ == '__main__':
    file_ = 'test'

    training_file = f'data/{file_}.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    index = 100

    image = X_train[index]
    print(y_train[index])

    plt.imshow(image)
    plt.show()

    image = preprocess(image)

    plt.imshow(image)
    plt.show()

    images = preprocess_images(X_train)
    save_images(images, f"data/{file_}")

    print(images.shape)
    print(images.dtype)

    plt.imshow(images[index])
    plt.show()
