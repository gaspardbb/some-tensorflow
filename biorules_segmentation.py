from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from matplotlib.widgets import Slider

from encoder_decoder import unet_with_functional_API
from tensorflow._api.v2.v2.keras import models
from tensorflow._api.v2.v2.keras.utils import plot_model
from time import time
import numpy as np

import matplotlib.pyplot as plt

from skimage import draw

"""
Image should have 2 channels.
First should be the nuclei.
Second should be the membrane.

This is a very simple example where I searched for what could be achieved if one used neural nets as what they 
are by essence: functions with a lot of parameters.

Here, I have a 2-channels image, which I want to segment. I'm looking for the segmentation which maximizes the 
difference between the inside and the outside. 

Of course, flow-based methods are so much more convenient to tackle this issue. 
Don't get me wrong: **the way I defined the loss there makes it preposterous** to use a neural network to solve it, 
since one may simply labelled each pixel comparing the value of its nuclei and membrane channel.

Yet, I find it's a good way to see neural networks in a different framework than usual. 
"""

# nuclei = np.load('resources/191ir.npy')[200:264, 200:264]
# membranes = np.load('resources/158gd.npy')[200:264, 200:264]
# nuclei = (nuclei - nuclei.min()) / (nuclei.max() - nuclei.min())
# membranes = (membranes - membranes.min()) / (membranes.max() - membranes.min())
# nuclei = nuclei * 2 - 1
# membranes = membranes * 2 - 1
# IMAGE = np.stack([nuclei, membranes], axis=-1)

IMAGE = np.zeros((64, 64, 2), dtype=np.float32)
target = draw.circle(32, 32, 16)
IMAGE[..., 0][target] = 1
IMAGE[..., 1] = 1
IMAGE[..., 1][target] = 0

IMAGE = IMAGE.reshape((1,) + IMAGE.shape)
print(f"Image shape is: {IMAGE.shape}")
IMAGE = tf.convert_to_tensor(IMAGE)


FILTERS_ENCODER = (16, 32, 64, 128, 256)
FILTERS_DECODER = (256, 128, 64, 32, 16)
N_CHANNELS_IN = 2
N_CHANNELS_OUT = 1
EPOCHS = 200
PRINT_RATE = 50
LAMBDA = .5
LEARNING_RATE = 7e-5
SAVE = True


segmentation_function = unet_with_functional_API(filters_encoder=FILTERS_ENCODER,
                                                 filters_decoder=FILTERS_DECODER,
                                                 n_channels_in=N_CHANNELS_IN,
                                                 n_channels_out=N_CHANNELS_OUT,
                                                 apply_batch_normalization=False,
                                                 apply_dropout=False,
                                                 # conv_params={'kernel_initializer': 'zeros'},
                                                 name='segmentation_function')

@tf.function
def segmentation_loss(inputs, outputs):
    """
The idea is to implement a loss which would give minimal value to a perfect segmentation on perfect channels. E.g,
putting 1 everywhere there's a nuclei and -1 everywhere there's a membrane would lead to a minimal score: nuclei
score would be low, and so would be the membranes score.

    Parameters
    ----------
    inputs
        image inputs. (1, n_rows, n_cols, n_channels). For now, n_channels=2.
    outputs
        output of the net.

    Returns
    -------
        The segmentation score
    """
    nuclei_score = - tf.reduce_mean(inputs[..., 0] * outputs)
    membranes_score = tf.reduce_mean(inputs[..., 1] * outputs)
    return nuclei_score, membranes_score


@tf.function
def train_step(f: models.Model, inputs, optimizer: tf.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        outputs = f(inputs)
        nuclei_score, membrane_score = segmentation_loss(inputs, outputs)
        penalty = LAMBDA * tf.abs(nuclei_score - membrane_score)
        # pixels_in = tf.reduce_sum(tf.where(outputs>0))
        # pixels_out = tf.reduce_sum(tf.where(outputs<=0))
        full_score = nuclei_score + membrane_score + penalty
    gradients = tape.gradient(full_score, f.trainable_variables)
    optimizer.apply_gradients(zip(gradients, f.trainable_variables))
    return {'nuclei_score': nuclei_score,
            'membrane_score': membrane_score,
            'penalty': penalty,
            'full_score': full_score}, outputs


score_types = ["nuclei_score", "membrane_score", "full_score", "penalty"]
optimizer = tf.optimizers.RMSprop(LEARNING_RATE)

metrics = {}
for score in score_types: metrics[score] = []

means = {}
for score in score_types: means[score] = tf.keras.metrics.Mean()

colors = ['r', 'g', 'b', 'y']
score_colors = {}
for i, score in enumerate(score_types): score_colors[score] = colors[i]


if SAVE:
    outputs = []
    to_plot = {}

to_format = "[{:>3d}/{:<3d}] ({} in {:.1f}s) pen: {penalty:8.2e} " \
            "nuc: {nuclei_score:8.2e} " \
            "mem: {membrane_score:8.2e} " \
            "ful: {full_score:8.2e}"

full_time = time()
t1 = time()
for epoch in range(EPOCHS):
    current_scores, current_output = train_step(segmentation_function, IMAGE, optimizer)

    if SAVE:
        outputs.append(current_output.numpy()[0, ..., 0])

    for score in score_types:
        metrics[score].append(current_scores[score])

    if (epoch != 0 and epoch % PRINT_RATE == 0) or epoch == 1 or epoch == EPOCHS-1:
        cur_time = time() - t1
        print(to_format.format(epoch, EPOCHS, PRINT_RATE, cur_time, **current_scores))
        t1 = time()

print(f"Did {EPOCHS} in {time()-full_time:.1f}s.")

result = segmentation_function(IMAGE)[0, ..., 0]

# ----------- Plot -----------
fig, axes = plt.subplots(1, 2)
for score in score_types:
    axes[0].plot(metrics[score], label=score, color=score_colors[score])

    if SAVE:
        to_plot[score], = axes[0].plot(0, metrics[score][0], "o", color=score_colors[score])

axes[0].legend()
im_result = axes[1].imshow(result, cmap='coolwarm')

# Animation loop
if SAVE:
    def update(index):
        index = int(index)
        im_result.set_data(outputs[index])
        im_result.set_clim(outputs[index].min(), outputs[index].max())

        for score in score_types:
            to_plot[score].set_xdata([index])
            to_plot[score].set_ydata([metrics[score][index]])

    slider = Slider(ax=fig.add_axes((.1, .03, .8, .02)), label="Image index", valmin=0,
                    valmax=len(outputs) - 1,
                    valinit=0, valfmt="%d", valstep=1)
    slider.on_changed(update)

plt.show()