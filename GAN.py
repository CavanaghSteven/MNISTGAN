import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from params import Params
import pandas as pd


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model_name, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fig_size = (100, 100)
        self.dim = (4, 4)
        self.mname = model_name

        self.num_examples = self.dim[0] * self.dim[1]
        np.random.seed(42)

        self.X = np.random.normal(0, 1, size=(self.num_examples, self.latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        gen_images = self.model.predict(self.X)

        gen_images = (0.5 * gen_images) + 0.5

        plt.figure(figsize=self.fig_size)
        for i in range(len(gen_images)):
            plt.subplot(self.dim[0], self.dim[1], i + 1)
            img = gen_images[i]
            plt.imshow(img)
            # plt.imshow(np.squeeze(gen_images[i], axis=2), cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
        plt.tight_layout()
        #         plt.savefig(f'data_dir/{self.mname}-{epoch}.png')
        plt.savefig(f'log/{self.mname}/imgs/{epoch}.png')
        plt.clf()
        plt.close()


class GAN():
    def __init__(self, discriminator, generator, params: Params):
        self.discriminator = discriminator
        self.generator = generator
        self.params = params
        self.log_fname = f'log/{params.model_name}/log.csv'

        # Instantiate one optimizer for the discriminator and another for the generator.
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        self.gp_weight = 10
        self.d_steps = 5

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.g_optimizer,
            discriminator_optimizer=self.d_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    @tf.function
    def augment_images(self, images, batch_size):

        images += 0.01 * tf.random.normal(tf.shape(images))

        # images = tf.image.resize(images, [self.IMG_DIM[0]+10, self.IMG_DIM[1]+10],
        #                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # images = tf.image.random_crop(images, size=[batch_size, *self.IMG_DIM, 3])
        images = tf.image.random_flip_left_right(images)

        return images

    @tf.function
    def d_loss_fn(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    @tf.function
    def g_loss_fn(self, fake_img):
        return -tf.reduce_mean(fake_img)

    @tf.function
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
    
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = (real_images * alpha) + ((1 - alpha) * fake_images)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.params.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors)

                real_images = self.augment_images(real_images, batch_size)
                fake_images = self.augment_images(fake_images, batch_size)

                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images)
                # Get the logits for real images
                real_logits = self.discriminator(real_images)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.params.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors)

            generated_images = self.augment_images(generated_images, batch_size)

            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {'d_loss': d_loss, 'g_loss': g_loss}

    #######################################################################################################

    def add_dicts(self, dict1, dict2):
        for key in dict1:
            if dict2[key].numpy() is np.nan:
                print('NAN Loss')
            dict1[key] += dict2[key].numpy()
        return dict1

    def divide_dict(self, dict, val):
        for key in dict:
            dict[key] /= val
        return dict

    def fit(self, dataset, dataset_length, callbacks: tf.keras.callbacks.CallbackList):

        initial_epoch = 0
        data_filename = f'log/{self.params.model_name}/history.csv'
        if os.path.exists(data_filename):
            data_df = pd.read_csv(data_filename)
            initial_epoch = int(np.array(data_df['epoch'])[-1] + 1)
            self.checkpoint.restore(
                tf.train.latest_checkpoint(f'log/{self.params.model_name}/{self.params.weight_dir_ext}/')
            )

        print('initial epoch:', initial_epoch)
        callbacks.set_model(self.generator)
        callbacks.on_train_begin()

        curr_iteration = 0
        for epoch_idx in range(initial_epoch, self.params.num_epochs):
            callbacks.on_epoch_begin(epoch_idx)
            total_losses = {'d_loss': 0, 'g_loss': 0}
            batch_progress_bar = tqdm(total=dataset_length)

            # for real_images in dataset:
            for real_images in dataset:
                callbacks.on_batch_begin(curr_iteration)
                # Train the discriminator & generator on one batch of real images.
                losses = self.train_step(real_images)
                total_losses = self.add_dicts(total_losses, losses)
                batch_progress_bar.set_description(
                    f'd_loss:{losses["d_loss"].numpy()}, g_loss:{losses["g_loss"].numpy()}'
                )
                batch_progress_bar.update(1)
                callbacks.on_batch_end(curr_iteration, losses)
                curr_iteration += 1

            # Checkpoint training after every epoch
            self.checkpoint.save(
                f'log/{self.params.model_name}/{self.params.weight_dir_ext}/'
            )

            batch_progress_bar.close()
            total_losses = self.divide_dict(total_losses, dataset_length)
            callbacks.on_epoch_end(epoch_idx, total_losses)

            # Shuffle data after every epoch
            dataset = dataset.unbatch()
            dataset = dataset.shuffle(buffer_size=1024).batch(self.params.batch_size)
