
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
from params import Params
from models import get_discriminator, get_generator
import cv2


def img_resize(img):
    return cv2.resize(img, (416, 416))


def save_frames_as_gif(frames, path='./log/', filename='animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    # plt.figure()

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='ffmpeg', fps=30)


params = Params()

d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
generator = get_generator(params)
discriminator = get_discriminator(params)

checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                 discriminator_optimizer=d_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(f'log/{params.model_name}/{params.weight_dir_ext}/-14')
# checkpoint.restore(tf.train.latest_checkpoint(f'log/{params.model_name}/{params.weight_dir_ext}/'))

num_random_points = 10
num_steps = 30
prev_point = np.random.normal(0, 1, size=params.latent_dim)
vectors = []

for point_idx in range(num_random_points):
    point = np.random.normal(0, 1, size=params.latent_dim)
    dist = np.linalg.norm(point - prev_point)
    ratios = np.linspace(0, 1, num=num_steps)
    interpolated_vectors = []
    for ratio in ratios:
        v = (1.0 - ratio) * prev_point + ratio * point
        interpolated_vectors.append(v)

    vectors.extend(np.asarray(interpolated_vectors))
    prev_point = point

vectors = np.array(vectors)
print('Shape of vectors', vectors.shape)
gen_images = generator.predict(vectors, verbose=1).astype('float32')
gen_images = (gen_images + 1.0) / 2.0
gen_images = list(map(img_resize, gen_images))
save_frames_as_gif(gen_images, path=f'log/{params.model_name}/')
