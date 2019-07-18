import tensorflow as tf
from tensorflow._api.v2.v2.keras import layers, models
from encoder_decoder import unet_with_functional_API, encoder
from tensorflow._api.v2.v2.keras.utils import plot_model
from time import time


def cycle_gan_model(x_input_shape, y_input_shape,
              filters_generator=((32, 64, 128, 256, 512), (256, 128, 64, 32)),
              filters_discriminator=(32, 64, 128)):

    filter_encoder, filter_decoder = filters_generator

    f_generator = unet_with_functional_API(filters_encoder=filter_encoder,
                                           filters_decoder=filter_decoder,
                                           n_channels_in=y_input_shape[-1],
                                           n_channels_out=x_input_shape[-1],
                                           apply_batch_normalization=True,
                                           apply_dropout=True,
                                           name='x_generator')

    g_generator = unet_with_functional_API(filters_encoder=filter_encoder,
                                           filters_decoder=filter_decoder,
                                           n_channels_in=x_input_shape[-1],
                                           n_channels_out=y_input_shape[-1],
                                           apply_batch_normalization=True,
                                           apply_dropout=True,
                                           name='y_generator')

    x_discriminator = encoder(filters_discriminator,
                              n_channels_in=x_input_shape[-1],
                              n_channels_out=1,
                              batch_normalization=(False,) + (True,) * (len(filters_discriminator) - 1),
                              name="x_discriminator"
                              )

    y_discriminator = encoder(filters_discriminator,
                              n_channels_in=y_input_shape[-1],
                              n_channels_out=1,
                              batch_normalization=(False,) + (True,) * (len(filters_discriminator) - 1),
                              name="y_discriminator"
                              )

    return f_generator, g_generator, x_discriminator, y_discriminator


@tf.function
def train_step(real_x, real_y, lambda_coeff, f_generator, g_generator, x_discriminator, y_discriminator,
               f_optimizer, g_optimizer, x_discriminator_optimizer, y_discriminator_optimizer):
    print('Tracing train_step...')

    # Compute variables
    with tf.GradientTape(persistent=True) as generator_tape, tf.GradientTape(persistent=True) as discriminator_tape:
        # F : Y -> X
        # G : X -> Y
        fake_x = f_generator(real_y)
        cycle_y = g_generator(fake_x)

        fake_y = g_generator(real_y)
        cycle_x = f_generator(fake_y)

        loss_discriminator_x = discriminator_loss(x_discriminator(real_x), x_discriminator(fake_x))
        loss_discriminator_y = discriminator_loss(y_discriminator(real_y), y_discriminator(fake_y))

        cycle_loss_x = cycle_loss(real_x, cycle_x)
        cycle_loss_y = cycle_loss(real_y, cycle_y)

        total_loss_discriminator = loss_discriminator_x + loss_discriminator_y
        total_loss_cycle = lambda_coeff * (cycle_loss_x + cycle_loss_y)

        total_loss = total_loss_discriminator + total_loss_cycle

    # Compute gradient
    generator_g_gradient = generator_tape.gradient(total_loss, g_generator.trainable_variables)
    generator_f_gradient = generator_tape.gradient(total_loss, f_generator.trainable_variables)

    discriminator_x_gradient = discriminator_tape.gradient(loss_discriminator_x, x_discriminator.trainable_variables)
    discriminator_y_gradient = discriminator_tape.gradient(loss_discriminator_y, y_discriminator.trainable_variables)

    # Apply gradients
    g_optimizer.apply_gradients(zip(generator_g_gradient, g_generator.trainable_variables))
    f_optimizer.apply_gradients(zip(generator_f_gradient, f_generator.trainable_variables))

    x_discriminator_optimizer.apply_gradients(zip(discriminator_x_gradient, x_discriminator.trainable_variables))
    y_discriminator_optimizer.apply_gradients(zip(discriminator_y_gradient, y_discriminator.trainable_variables))

    # return metrics
    return {'total': total_loss, 'cycle': total_loss_cycle, 'discriminator': total_loss_discriminator}


def train_cycle_gan(g_generator, f_generator, x_discriminator, y_discriminator, g_generator_optimizer,
f_generator_optimizer, x_discriminator_optimizer, y_discriminator_optimizer, x_ds, y_ds,
        lambda_coeff, batch_size, steps_per_epoch, epochs,):

    checkpoint_path = "checkpoints/LAMBDA%s_BS%s" % (lambda_coeff, batch_size)
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
        print('Latest checkpoint restored!!')

    metrics = {'total': [],
               'discriminator': [],
               'cycle': []}

    # For multiple GPU handling
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # n_gpus = len(gpus)
    # splits = [STEPS_PER_EPOCH // n_gpus] * n_gpus
    # for i in range(STEPS_PER_EPOCH % n_gpus):
    #     splits[i] += 1

    full_time = time()
    for epoch in range(epochs):
        total_loss_metric = tf.keras.metrics.Mean()
        discriminator_loss_metric = tf.keras.metrics.Mean()
        cycle_loss_metric = tf.keras.metrics.Mean()

        time_epoch = time()
        # if gpus:
        #     for load, gpu in zip(splits, gpus):
        #         with tf.device(gpu.name):
        #             for ellipse, cell in tf.data.Dataset.zip((ellipse_ds, cells_ds)).take(load):
        #
        #
        # else:
        for ellipse, cell in tf.data.Dataset.zip((x_ds, y_ds)).take(steps_per_epoch):
            losses = train_step(cell, ellipse, lambda_coeff,
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
        print(f"{epoch + 1:3d}/{epochs:<3d} [{time() - time_epoch:6.2f} s]"
              f" tot: {metrics['total'][-1]:5.2e} "
              f"dis: {metrics['discriminator'][-1]:5.2e} "
              f"cyc: {metrics['cycle'][-1]:5.2e}")

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path), end=' ')

    print(f'Did {epochs} in {time() - full_time: .2f}s.')

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    fake_loss = loss_obj(tf.zeros_like(generated), generated)
    return .5 * (real_loss + fake_loss)


def cycle_loss(image, cycled_image):
    return tf.reduce_mean(tf.abs(image - cycled_image))