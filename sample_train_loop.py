from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from skimage.draw import circle as draw_circle
from skimage.util.noise import random_noise
from time import time

from cvae import VAE, train_step

STEPS_PER_EPOCHS = 10
EPOCHS = 1
BATCH_SIZE = 10


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