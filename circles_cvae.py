from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from skimage.draw import circle as draw_circle
from skimage.util.noise import random_noise
from time import time

from cvae import VAE, train_step

class CircleGenerator:

    def __init__(self, image_shape, radius_center, radius_circle):
        self.radius_circle = radius_circle
        self.radius_center = radius_center
        self.image_shape = image_shape

    def get_one_sample(self, noise=None):
        result = np.zeros(self.image_shape, dtype=np.float32)
        theta = np.random.random() * 2 * np.pi
        x, y = self.radius_center * np.cos(theta), self.radius_center * np.sin(theta)
        x += self.image_shape[0] // 2
        y += self.image_shape[1] // 2
        result[draw_circle(x, y, self.radius_circle, shape=self.image_shape)] = 1

        if noise is not None:
            result = random_noise(result, noise)

        result = result[..., np.newaxis]

        return result

    def generator(self, noise=None):
        def circle_generator():
            while True:
                result = self.get_one_sample(noise)
                yield result
        return circle_generator


if __name__ == '__main__':
    IMAGE_SHAPE = (64, 64)
    BATCH_SIZE = 100
    EPOCHS = 3
    STEPS_PER_EPOCH = 10
    N_SAMPLES = 1
    LOG_SIGMA = -3.

    vae = VAE(filters_encoder=(32, 64, 128, 256),
              filters_decoder=(256, 64, 128, 32),
              latent_dim=100)
    vae.build((BATCH_SIZE, ) + IMAGE_SHAPE + (1,))
    optimizer = tf.optimizers.RMSprop(learning_rate=1e-4)

    circle_generator = CircleGenerator(IMAGE_SHAPE, 16, 12)
    generator = circle_generator.generator()
    dataset = tf.data.Dataset.from_generator(generator, tf.float32, IMAGE_SHAPE + (1,)).batch(BATCH_SIZE)

    checkpoint_path = "checkpoints/LSIGMA%s_BS%s" % (LOG_SIGMA, BATCH_SIZE)
    ckpt = tf.train.Checkpoint(vae=vae,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    metrics = {'total': [],
               'kl_divergence': [],
               'recon_loss': []}

    full_time = time()
    for epoch in range(EPOCHS):
        total_loss_metric = tf.keras.metrics.Mean()
        kl_div_loss_metric = tf.keras.metrics.Mean()
        recon_loss_metric = tf.keras.metrics.Mean()

        time_epoch = time()
        for circle in dataset.take(STEPS_PER_EPOCH):
            losses = train_step(circle, vae, optimizer, n_samples=N_SAMPLES, log_sigma=LOG_SIGMA)
            total_loss_metric(losses['total'])
            kl_div_loss_metric(losses['kl_divergence'])
            recon_loss_metric(losses['recon_loss'])
        metrics['total'].append(total_loss_metric.result())
        metrics['kl_divergence'].append(kl_div_loss_metric.result())
        metrics['recon_loss'].append(recon_loss_metric.result())
        print(f"{epoch + 1:3d}/{EPOCHS:<3d} [{time() - time_epoch:6.2f} s]"
              f"tot: {metrics['total'][-1]:5.2e} "
              f"div: {metrics['kl_divergence'][-1]:5.2e} "
              f"rec: {metrics['recon_loss'][-1]:5.2e}")

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path), end=' ')

    print(f'Did {EPOCHS} in {time() - full_time: .2f}s.')


# arr = circle_generator.get_one_sample()
# res = vae(arr[np.newaxis, ...])
#
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(arr[..., 0])
# axes[1].imshow(res[0, 0, ..., 0])
# plt.show()