
from util import create_dir
from GAN import GANMonitor, GAN
from data import get_dataset
from params import Params
from models import get_discriminator, get_generator
import tensorflow as tf

if __name__ == '__main__':

    params = Params()

    create_dir('log')
    create_dir(f'log/{params.model_name}')
    create_dir(f'log/{params.model_name}/{params.weight_dir_ext}')
    create_dir(f'log/{params.model_name}/{params.imgs_dir_ext}')

    dataset, dataset_length = get_dataset(params.batch_size)

    # Create models
    gen = get_generator(params)
    dis = get_discriminator(params)
    eval_obj = GANMonitor(params.model_name, params.latent_dim)
    gan = GAN(dis, gen, params)

    csvlogger = tf.keras.callbacks.CSVLogger(
        f'log/{params.model_name}/history.csv',
        separator=',',
        append=True)

    tensorboard = tf.keras.callbacks.TensorBoard(
        f'log/{params.model_name}',
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )

    callbacks = tf.keras.callbacks.CallbackList([
        csvlogger,
        tensorboard,
        eval_obj
    ])

    # Train models
    gan.fit(
        dataset,
        dataset_length,
        callbacks
    )
