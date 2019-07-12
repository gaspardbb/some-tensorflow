from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from skimage.draw import ellipse as draw_ellipse
from time import time
from tensorflow._api.v2.v2 import keras

from cycle_gan import cycle_gan_model, train_step

import os

# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.1/lib64/'


"""
Here's a try to use Keras fit and compile function with the Cycle GAN. 
We just saw that we struggled to define a cycle gan with the functional API, as we had to handle multiple outputs.
Here, the `fit` call does not work, on the datasets, and once we make it work it throws an error relative to the 
gradients. Yet, it's nearly impossible to trace back that error, or at least to my knowledge. 

That's why I reverted to manually writing the training loop. I had bugs first (gradients not defined) but I could 
easily corrected them by looking at the parts of the code which failed. 

CONCLUSION: for models with multiple inputs/outputs which interact together, I think I'll use a manual training loop.
"""


OUTPUT_CHANNEL = 1
CROP_SIZE = (64, 64)
OUTPUT_SHAPE = CROP_SIZE + (OUTPUT_CHANNEL,)
BATCH_SIZE = 400
NUM_ELLIPSE = (10, 20)
LAMBDA = 100
EPOCHS = 1
STEPS_PER_EPOCH = 10


def add_ellipse(array):
    n_rows, n_cols = array.shape
    r, c = np.random.normal(5, 3, 2)
    x, y = np.random.random(2) * np.array([n_rows, n_cols])
    rotation = (np.random.random() - .5) * 2 * np.pi
    array[draw_ellipse(x, y, r, c, shape=array.shape, rotation=rotation)] = 1


def ellipse_generator(num_ellipse=NUM_ELLIPSE, shape=CROP_SIZE):
    min_ellipse, high_ellipse = num_ellipse
    while True:
        array = np.zeros(shape, dtype=np.dtype('float32'))
        num_ellipse = np.random.randint(min_ellipse, high_ellipse)
        for _ in range(num_ellipse):
            add_ellipse(array)
        array = np.reshape(array, OUTPUT_SHAPE)
        #     array = tf.convert_to_tensor(array, dtype=tf.dtypes.float32)
        #     array = tf.reshape(array, OUTPUT_SHAPE)
        yield array


train_array = [np.load('resources/191_interface_unequalized.npy'),
               np.load('resources/191_GBM_unequalized.npy')]
train_tensors = [tf.convert_to_tensor(array, dtype=tf.dtypes.float32) for array in train_array]
train_tensors = [tf.reshape(tensor, tensor.shape.as_list() + [OUTPUT_CHANNEL]) for tensor in train_tensors]


# @tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.dtypes.int8)])
# Generator are not supported by AutoGraph yet!
def cells_generator(shape=OUTPUT_SHAPE):
    n_samples = len(train_tensors)
    while True:
        train_tensor = train_tensors[np.random.randint(0, n_samples)]
        return_tensor = tf.image.random_crop(train_tensor, size=shape)
        return_tensor = tf.image.random_flip_left_right(tf.image.random_flip_up_down(return_tensor))
        return_tensor = tf.image.random_brightness(tf.image.random_contrast(return_tensor, 0, 1), max_delta=1)
        return_tensor = (return_tensor - tf.reduce_min(return_tensor)) / tf.reduce_max(return_tensor)
        yield return_tensor


ellipse_ds = tf.data.Dataset.from_generator(ellipse_generator,
                                            output_types=tf.dtypes.float32,
                                            output_shapes=OUTPUT_SHAPE).batch(BATCH_SIZE)
cells_ds = tf.data.Dataset.from_generator(cells_generator,
                                          output_types=tf.dtypes.float32,
                                          output_shapes=OUTPUT_SHAPE).batch(BATCH_SIZE)


f_generator, g_generator, x_discriminator, y_discriminator = cycle_gan_model(OUTPUT_SHAPE, OUTPUT_SHAPE)

g_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
f_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

x_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
y_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "checkpoints/LAMBDA%s_BS%s" % (LAMBDA, BATCH_SIZE)
ckpt = tf.train.Checkpoint(g_generator=g_generator,
                           f_generator=f_generator,
                           x_discriminator=x_discriminator,
                           y_discriminator=y_discriminator,
                           g_generator_optimizer=g_generator_optimizer,
                           f_generator_optimizer=f_generator_optimizer,
                           x_discriminator_optimizer=x_discriminator_optimizer,
                           y_discriminator_optimizer=y_discriminator_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


metrics = {'total': [],
           'discriminator': [],
           'cycle': []}

full_time = time()
for epoch in range(EPOCHS):
    total_loss_metric = tf.keras.metrics.Mean()
    discriminator_loss_metric = tf.keras.metrics.Mean()
    cycle_loss_metric = tf.keras.metrics.Mean()

    time_epoch = time()
    for ellipse, cell in tf.data.Dataset.zip((ellipse_ds, cells_ds)).take(STEPS_PER_EPOCH):
        losses = train_step(cell, ellipse, LAMBDA,
                            f_generator=f_generator, g_generator=g_generator,
                            x_discriminator=x_discriminator, y_discriminator=y_discriminator,
                            f_optimizer=f_generator_optimizer, g_optimizer=g_generator_optimizer,
                            x_discriminator_optimizer=x_discriminator_optimizer,
                            y_discriminator_optimizer=y_discriminator_optimizer)
        total_loss_metric(losses['total'])
        discriminator_loss_metric(losses['discriminator'])
        cycle_loss_metric(losses['cycle'])
    metrics['total'].append(total_loss_metric.result())
    metrics['discriminator'].append(discriminator_loss_metric.result())
    metrics['cycle'].append(cycle_loss_metric.result())
    print(f"{epoch + 1:3d}/{EPOCHS:<3d} [{time() - time_epoch:6.2f} s]"
          f" tot: {metrics['total'][-1]:5.2e} "
          f"dis: {metrics['discriminator'][-1]:5.2e} "
          f"cyc: {metrics['cycle'][-1]:5.2e}")

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path), end=' ')

print(f'Did {EPOCHS} in {time() - full_time: .2f}s.')


# WITH FUNCTIONAL API
# model = cycle_gan_model(x_input_shape=OUTPUT_SHAPE, y_input_shape=OUTPUT_SHAPE)
#
# callbacks = [
#     keras.callbacks.EarlyStopping(
#         monitor='cycle_loss',
#         min_delta=1e-2,
#         patience=5,
#         verbose=1),
#     keras.callbacks.ModelCheckpoint(
#         filepath='./models',
#         save_best_only=True
#     )]
#
# model.compile(optimizer=keras.optimizers.RMSprop(.01),
#               loss={'x_real_cycle': gan_cycle_loss,
#                     'y_real_cycle': gan_cycle_loss,
#                     'x_disc_real_fake': gan_discriminator_loss,
#                     'y_disc_real_fake': gan_discriminator_loss},
#               loss_weight={'x_real_cycle': LAMBDA / 2,
#                            'y_real_cycle': LAMBDA / 2,
#                            'x_disc_real_fake': 1,
#                            'y_disc_real_fake': 1})
#
#
# def all_generator():
#     gen_cell = cells_generator()
#     gen_ellipse = ellipse_generator()
#     while True:
#         yield ({'x_input': next(gen_cell),
#                'y_input': next(gen_ellipse)}, )
#
# all_dataset = tf.data.Dataset.from_generator(all_generator, output_types=({'x_input': tf.dtypes.float32,
#                                                                           'y_input': tf.dtypes.float32},),
#                                              output_shapes=({'x_input': OUTPUT_SHAPE,
#                                                             'y_input': OUTPUT_SHAPE},)).batch(2)
#
# model.fit(all_dataset)