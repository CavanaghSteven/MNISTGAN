import numpy as np
import tensorflow as tf
from params import Params


def resize_image(img):
    return tf.image.resize(img, Params().img_dim[:2])


def get_dataset(batch_size):
    # the data, split between train and test sets
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    all_digits = np.concatenate([x_train, x_test])

    # Scale images to the [-1, 1] range
    all_digits = (all_digits.astype("float32") / 127.5) - 1.0
    # Make sure images have shape (28, 28, 1)
    all_digits = np.expand_dims(all_digits, -1)

    # print('\n', np.shape(x_train), np.shape(x_test), np.shape(all_digits), '\n')

    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    # dataset = dataset.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.cache('dataset_cache')
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_length = len(all_digits) // batch_size

    return dataset, dataset_length
