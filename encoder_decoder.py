import tensorflow as tf
from tensorflow._api.v2.v2.keras import layers, models

"""
We want to implement the a downsample block. If we do not wish to specify the input shape, we cannot use the 
functional API, which requires an input layer with the right number of channel: 
```python
    inputs = layers.Input(shape=[None, None, None])
    outputs = Conv2D(1, 3)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs)
```
raises:
```
    ValueError: The channel dimension of the inputs should be defined. Found `None`.
```

If we're building a UNet, it's tiring to check the input shape of every blocks. Well, it's not that complicated:
    * For the encoder : filters_encoder[i-1]
    * For the decoder : filters_encoder[i] + filters_decoder[i-1]
But we may make errors, and it soon gets too complex when dealing with more complicated networks.  

One thing we can do is use the Sequential Model, which has a specific `build` method which will create such an 
'Input' layer. 
"""


def downsample_with_sequential(filters, size, apply_batchnorm=True, name=None):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


"""
If we want something more general, we can subclass the Layer class. Let's note that we could define all the layers in 
the __init__ method.
"""


class Downsample(layers.Layer):
    """
    A downsample block consisting of a strided Conv2D layer, possibly followed by a BatchNormalization layer,
    followed by a LeakyReLU activation.
    """

    def __init__(self, filters: int, kernel_size: int, batch_normalization: bool = True, name: str = None, **kwargs):
        """

        Parameters
        ----------
        filters : int
            The number of filters of the convolution layer.
        kernel_size : int
            The size of the kernel for the convolution layer.
        batch_normalization : bool
            Whether to apply batch normalization.
        name : str
            Name of the block.
        kwargs : dict
            Other Parameters applied to the parent's constructor.
        """
        super(Downsample, self).__init__(name=name, **kwargs)
        self.apply_batch_normalization = batch_normalization
        self.kernel_size = kernel_size
        self.filters = filters

    def build(self, input_shape):
        self.strided_convolution = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                                 strides=2, padding='same', input_shape=input_shape)
        if self.apply_batch_normalization:
            self.batch_normalization = layers.BatchNormalization()

        self.activation = layers.LeakyReLU()

    def call(self, inputs, **kwargs):
        x = self.strided_convolution(inputs)
        if self.apply_batch_normalization:
            x = self.batch_normalization(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(Downsample, self).get_config()
        config.update({'apply_batch_normalization': self.apply_batch_normalization,
                       'kernel_size': self.kernel_size,
                       'filters': self.filters})
        return config


class Upsample(layers.Layer):
    """
    A transposed Conv2D, possibly followed by a dropout, with a LeakyReLU activation.
    """

    def __init__(self, filters: int, kernel_size: int, dropout: bool = True, name: str = None, **kwargs):
        super(Upsample, self).__init__(name=name, **kwargs)
        self.apply_dropout = dropout
        self.kernel_size = kernel_size
        self.filters = filters

    def build(self, input_shape):
        super(Upsample, self).build(input_shape)
        self.transposed_convolution = layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size,
                                                             strides=2, padding='same', input_shape=input_shape)
        if self.apply_dropout:
            self.dropout = layers.Dropout(.5)
        self.activation = layers.LeakyReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.transposed_convolution(inputs)
        if training:
            # tf.cond executes *all* branches. `if training and self.apply_dropout` won't work here.
            if self.apply_dropout:
                x = self.dropout(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({'apply_dropout': self.apply_dropout,
                       'kernel_size': self.kernel_size,
                       'filters': self.filters})
        return config


"""
Now we want to define the UNet model. We can use the subclass API, but it's not the easiest way to do. Still, 
if later we want to implement specific losses, it can be a good way to understand what's going on. Note that we DO 
NOT need to specify the input_shape. 
"""


class UNet(models.Model):

    def __init__(self, filters_encoder, filters_decoder, n_channels_out,
                 apply_batch_normalization, apply_dropout,
                 kernel_size_encoder=3, kernel_size_decoder=3,
                 name=None, **kwargs):
        super(UNet, self).__init__(name=name, **kwargs)

        # Check length of encoding and decoding paths
        assert len(filters_encoder) == len(filters_decoder) + 1

        # Turn kernel_sizes in list if they are int
        if type(kernel_size_encoder) is int:
            kernel_size_encoder = [kernel_size_encoder] * len(filters_encoder)
        if type(kernel_size_decoder) is int:
            kernel_size_decoder = [kernel_size_decoder] * len(filters_decoder)

        self.encoder_stack = [Downsample(n_filters, kernel_size, apply_batch_normalization)
                              for n_filters, kernel_size in zip(filters_encoder, kernel_size_encoder)]
        self.decoder_stack = [Upsample(n_filters, kernel_size, apply_dropout)
                              for n_filters, kernel_size in zip(filters_decoder, kernel_size_decoder)]

        self.last_layer = layers.Conv2DTranspose(filters=n_channels_out, kernel_size=3, strides=2,
                                                 padding='same', activation='tanh')
        self.concatenate = layers.concatenate

    # def build(self, input_shape):
    #     assert input_shape[0] // 2 ** len(self.encoder_stack)
    #     assert input_shape[1] // 2 ** len(self.encoder_stack)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        skips = []
        for enc in self.encoder_stack:
            x = enc(x)
            skips.append(x)

        for skip, dec in zip(skips[-2::-1], self.decoder_stack):
            x = dec(x)
            x = self.concatenate([x, skip])

        x = self.last_layer(x)

        return x


"""
Otherwise, we can simply use the functional API. It's easier to write, but it comes at a cost; we need to specify the 
number of channels of the input.  
"""


def unet_with_functional_API(filters_encoder, filters_decoder, n_channels_in, n_channels_out,
                             apply_batch_normalization, apply_dropout,
                             kernel_size_encoder=3, kernel_size_decoder=3,
                             name=None, **kwargs):
    if type(kernel_size_encoder) is int:
        kernel_size_encoder = [kernel_size_encoder] * len(filters_encoder)
    if type(kernel_size_decoder) is int:
        kernel_size_decoder = [kernel_size_decoder] * len(filters_decoder)

    encoder_stack = [Downsample(n_filters, kernel_size, apply_batch_normalization)
                     for n_filters, kernel_size in zip(filters_encoder, kernel_size_encoder)]
    decoder_stack = [Upsample(n_filters, kernel_size, apply_dropout)
                     for n_filters, kernel_size in zip(filters_decoder, kernel_size_decoder)]

    skips = []

    inputs = layers.Input(shape=[None, None, n_channels_in])
    x = inputs

    for enc in encoder_stack:
        x = enc(x)
        skips.append(x)

    for skip, dec in zip(skips[-2::-1], decoder_stack):
        x = dec(x)
        x = layers.Concatenate()([x, skip])

    outputs = layers.Conv2DTranspose(filters=n_channels_out, kernel_size=3, strides=2,
                                     padding='same', activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)


def encoder(encoder_filters, n_channels_in, n_channels_out, batch_normalization, kernel_size=3,
            name=None, **kwargs):
    if type(kernel_size) is int:
        kernel_size = [kernel_size] * len(encoder_filters)

    if type(batch_normalization) is bool:
        batch_normalization = [batch_normalization] * len(encoder_filters)

    assert len(encoder_filters) == len(kernel_size) == len(batch_normalization)

    encoder_stack = [Downsample(filters, size, batch_norm)
                     for filters, size, batch_norm in zip(encoder_filters, kernel_size, batch_normalization)]

    inputs = layers.Input(shape=[None, None, n_channels_in])

    x = inputs
    for enc in encoder_stack:
        x = enc(x)

    outputs = layers.Conv2D(n_channels_out, 3, padding='same', activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)


def decoder(decoder_filters, n_channels_in, n_channels_out, dropout, kernel_size=3,
            name=None, **kwargs):
    if type(kernel_size) is int:
        kernel_size = [kernel_size] * len(decoder_filters)

    if type(dropout) is bool:
        dropout = [dropout] * len(decoder_filters)

    assert len(decoder_filters) == len(kernel_size) == len(dropout)

    decoder_stack = [Upsample(filters, size, apply_dropout)
                     for filters, size, apply_dropout in zip(decoder_filters, kernel_size, dropout)]

    inputs = layers.Input(shape=[None, None, n_channels_in])

    x = inputs
    for dec in decoder_stack:
        x = dec(x)

    outputs = layers.Conv2D(n_channels_out, 3, padding='same', activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)


if __name__ == '__main__':
    a = tf.zeros((1, 64, 64, 3))
    b = tf.zeros((1, 64, 64, 5))

    model_api = unet_with_functional_API((16, 32, 64, 128), (64, 32, 16), 3, 5, True, True)
    model = UNet((10, 5), (8,), 5, True, True)
    model_encoder = encoder((16, 32, 64), 3, 1, [True] * 3)
    model_decoder = decoder((16, 24, 32), 12, 1, False)
