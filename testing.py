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

    predictions: np.ndarray = new_model.predict(tf.convert_to_tensor(image), batch_size=64)

    N = 5

    top_n_predictions = tf.nn.top_k(predictions, k=N)

    print(top_n_predictions.values)
    print()
    print(np.around(top_n_predictions.values.numpy() * 100, 1))

    string_predictions = np.vectorize(lambda index: signs[index])(top_n_predictions.indices.numpy())

    print(string_predictions)
