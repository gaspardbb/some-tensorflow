import tensorflow as tf
from tensorflow._api.v2.v2.keras import layers, models
from encoder_decoder import encoder, decoder
import numpy as np


@tf.function
def vae_kl_divergence(mean: tf.Tensor, logvar: tf.Tensor):
    """
    Compute the KL divergence of a independent multidimensional distribution w.r.t a centered-scaled normal.

    Parameters
    ----------
    mean
        Vector of mean.
    logvar
        Vector of log variance.

    Returns
    -------
    The KL divergence between the two values

The constant terms are added, even though they do not change the gradient descent. Full formula is:
\mathcal{D}[\mathcal{N}(\mu_0,\Sigma_0) \| \mathcal{N}(\mu_1,\Sigma_1)] = \frac{ 1 }{ 2 } \left( \mathrm{tr} \left( \Sigma_1^{-1} \Sigma_0 \right) + \left( \mu_1 - \mu_0\right)^\top \Sigma_1^{-1} ( \mu_1 - \mu_0 ) - k + \log \left( \frac{ \det \Sigma_1 }{ \det \Sigma_0  } \right)  \right)
    """
    print("Tracing vae_kl_divergence...")

    n_dim = mean.shape[0]
    norm_mean = tf.pow(tf.linalg.norm(mean), 2)
    trace = tf.reduce_sum(tf.exp(logvar))
    log_det = tf.reduce_sum(logvar)
    return .5 * (trace - log_det + norm_mean - n_dim)


@tf.function
def reconstruction_loss(inputs, outputs, n_samples, log_sigma, n_dim):
    """
Reconstruction loss for a VAE, handling Monte Carlo sampling with more than 1 sample.

    Parameters
    ----------
    inputs
        Should have shape: (batch_size, None, None, n_channels)
    outputs
        Should have shape: (batch_size, n_samples, None, None, n_channels)
    n_samples : int
        The number of samples for the Monte Carlo. Could be guessed from outputs.
    log_sigma : float
        The standard deviation of P(X | z) ~ N( f(z) ; sigma^2 I )
    n_dim
        The number of latent dimension
    """
    tf.assert_equal(inputs.shape[0], outputs.shape[0])
    tf.assert_equal(inputs.shape[1:], outputs.shape[2:])
    tf.assert_equal(inputs.shape[1], n_samples)

    batch_size, _, n_rows, n_cols, n_channels = outputs.shape[0]

    inputs = tf.reshape(inputs, (batch_size, 1, n_rows, n_cols, n_channels))

    difference = tf.pow(inputs - outputs, 2)
    # Shape is: (batch_size, n_samples, n_rows, n_cols, n_channels)

    pixels_l2_loss = tf.reduce_sum(difference, axis=[2, 3, 4])
    # Shape is : (batch_size, n_samples)

    monte_carlo = tf.reduce_mean(pixels_l2_loss, axis=1)
    # Shape is : (batch_size, )

    constants = - 2 * n_dim * log_sigma - n_dim / 2 * tf.math.log(2 * np.pi)

    return - 1 / (2 * tf.exp(2 * log_sigma)) * monte_carlo + constants



class TestLoss(models.Model):

    def call(self, inputs, **kwargs):
        loss_max = tf.reduce_max(inputs)
        loss_min = tf.reduce_min(inputs)
        tf.print(loss_max)
        tf.print(loss_min)
        self.add_metric(loss_max, aggregation="mean", name="max")
        self.add_metric(loss_min, aggregation="mean", name="min")
        return tf.reduce_sum(inputs)


# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# model = TestLoss()
# model(a)
# model(b)
# print(f'With shape {a.shape} : {[model.metrics[i].result() for i in [0, 1]]}')
# a = tf.reshape(a, (1, 2, 2))
# model(a)
# print(f'With shape {b.shape} : {[model.metrics[i].result() for i in [0, 1]]}')


class VAE(models.Model):

    def __init__(self, filters_encoder, filters_decoder, latent_dim):
        super(VAE, self).__init__()
        assert len(filters_decoder) == len(filters_encoder)

        self.latent_dim = latent_dim
        self.filters_decoder = filters_decoder
        self.filters_encoder = filters_encoder

        self.flatten = layers.Flatten(name="encoder_to_flatten")
        self.to_latent = layers.Dense(latent_dim * 2, activation=None, name="encoder_to_latent_space")

    def build(self, input_shape):
        n_channels = input_shape[-1]

        contracting_power = 2**(len(self.filters_encoder) - 1)
        assert input_shape[1] % contracting_power == 0
        assert input_shape[2] % contracting_power == 0

        # Dimension at the end of the encoding part
        n_rows, n_cols = (input_shape[1] // contracting_power,
                          input_shape[2] // contracting_power)

        self.from_latent = layers.Dense(n_rows * n_cols * self.filters_decoder[0])
        self.reshape_from_latent = layers.Reshape((n_rows, n_cols, self.filters_decoder[0]))

        # Input shape is : [None, None, None, n_channels]
        self.encoder_net = encoder(self.filters_encoder[:-1], n_channels_in=n_channels,
                                   n_channels_out=self.filters_encoder[-1], batch_normalization=False,
                                   name="encoder")
        # Output shape is : [None, None, None, self.filters_encoder[-1]]

        # Input shape is : [None, None, None, self.filters_decoder[0]]
        self.decoder_net = decoder(self.filters_decoder[1:], n_channels_in=self.filters_decoder[0],
                               n_channels_out=n_channels, dropout=False, name="decoder")
        # Output shape is : [None, None, None, n_channels]

    @tf.function
    def encode(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        # Input shape is : [batch_size, None, None, n_channels]
        x = self.encoder_net(inputs)
        x = self.flatten(x)
        x = self.to_latent(x)
        # x is now a vector of size: (batch_size, 2*latent_dim)
        mean, logvar = tf.split(x, 2, axis=1)

        # Output shape is : [batch_size, latent_dim], [batch_size, latent_dim]
        return mean, logvar

    @staticmethod
    @tf.function
    def reparameterize(mean, log_var, n_samples):
        # Input shape is : [batch_size, latent_dim], [batch_size, latent_dim]
        batch_size, latent_dim = mean.shape

        # We do this combination of tile and reshape to put the added samples on the 0-axis, the one usually
        # corresponding to the batch size. It's strictly equivalent to using the numpy 'repeat' function, but it does
        # not exist for tensorflow. Check the end of this file for example.
        mean = tf.tile(mean, (1, n_samples))
        mean = tf.reshape(mean, (batch_size * n_samples, latent_dim))
        log_var = tf.tile(log_var, (1, n_samples))
        log_var = tf.reshape(log_var, (batch_size * n_samples, latent_dim))
        epsilon = tf.random.normal(shape=[batch_size * n_samples, latent_dim])

        # Output shape is : [batch_size, latent_dim]
        return epsilon * tf.exp(log_var * .5) + mean

    def samples(self, n_samples=1, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decode(epsilon, apply_sigmoid=True)

    @tf.function
    def decode(self, z_sample, apply_sigmoid=False):
        # Input shape is : [batch_size, latent_dim]
        tf.assert_equal(self.built, True, message="You need to build your net prior to doecoding something!")
        x = self.from_latent(z_sample)
        x = self.reshape_from_latent(x)
        x = self.decoder_net(x)

        if apply_sigmoid:
            x = tf.sigmoid(x)

        # Output shape is : [batch_size, None, None, n_channels]
        return x

    def call(self, inputs, n_samples=1, **kwargs):
        batch_size = inputs.shape[0]
        n_rows, n_cols, n_channels = inputs.shape[1:]
        # Shape is : (batch_size, None, None, n_channels)
        mean, logvar = self.encode(inputs)
        # Shape is : (batch_size, latent_dim)
        z_samples = self.reparameterize(mean, logvar, n_samples)
        # Shape is : (batch_size * n_samples, latent_dim)
        outputs = self.decode(z_samples, apply_sigmoid=True)
        # Shape is : (batch_size * n_samples, None, None, n_channels)
        outputs = tf.reshape(outputs, (batch_size, n_samples, n_rows, n_cols, n_channels))
        # Shape is : (batch_size, n_samples, None, None, n_channels)
        return outputs


a = tf.constant(1, shape=(5, 64, 64, 3), dtype=tf.dtypes.float32)  # batch_size of 5, 64x64 RGB image
model = VAE(filters_encoder=(10, 20, 30), filters_decoder=(30, 20, 10), latent_dim=8)
model(a, n_samples=11)

"""
Here's to check that the transformation I do for the MC is correct.
Let's say we have a batch_size of 3, and latent_dim of 5. We have mean :

mean = tf.convert_to_tensor([[ 0,  1,  2,  3,  4],
                          [ 5,  6,  7,  8,  9],
                          [10, 11, 12, 13, 14]])

We want n_samples=4. We first tile along axis 1.

mean = tf.tile(mean, (1, 4))
>>>     [[ 0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14]]

Then we reshape, to have shape: batch_size * n_samples, latent_dim:

mean = tf.reshape(mean, (3*4, 5))
>>>   [[ 0,  1,  2,  3,  4],
       [ 0,  1,  2,  3,  4],
       [ 0,  1,  2,  3,  4],
       [ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [ 5,  6,  7,  8,  9],
       [ 5,  6,  7,  8,  9],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [10, 11, 12, 13, 14],
       [10, 11, 12, 13, 14],
       [10, 11, 12, 13, 14]]
       
And we have the result we want. It will propagate nicely through the rest of the network, because what's added is on 
the first dimension, the batch size. We'll only have to add a dimension by reshaping to: [batch_size, n_samples, 
...] in the end.

"""