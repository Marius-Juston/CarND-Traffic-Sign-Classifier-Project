import os

import cv2
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    signs = np.genfromtxt('signnames.csv', delimiter=',', names=True, dtype=[np.int8, 'U50'])['SignName']
    print(signs)

    image_shape = (32, 32, 3)
    image_folder = 'real_images'

    image = np.array(
        [cv2.resize(

            cv2.cvtColor(
                cv2.imread(f'{image_folder}/{filename}'),
                cv2.COLOR_BGR2RGB),
            image_shape[:-1], interpolation=cv2.INTER_LINEAR)

            for filename in os.listdir(image_folder)])

    new_model = tf.keras.models.load_model('model20200530-211606/')
    indices = np.argmax(new_model.predict(tf.convert_to_tensor(image), batch_size=64), axis=1)

    print(np.fromiter((signs[xi] for xi in indices), 'U50'))
