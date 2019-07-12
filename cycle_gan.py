import tensorflow as tf
from tensorflow._api.v2.v2.keras import layers, models
from encoder_decoder import unet_with_functional_API, encoder, discriminator_loss, cycle_loss
from tensorflow._api.v2.v2.keras.utils import plot_model


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