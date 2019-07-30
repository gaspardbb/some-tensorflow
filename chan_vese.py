from time import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation._chan_vese import _cv_init_level_set


"""
This is a implementation of the Chan Vese segmentation flow on Tensorflow.  
A run on *CPU*, with an image of size 500x500 made it 4 times faster than sklearn's implementation.
On 100x100 image, sklearn's implementation is slightly faster: the overhead induced by building the graph makes it 
less interesting.  
"""

"""
At first I thought it would be necessary to specify the input shape in the tf.function, to avoid retracing at each 
call of the function. In fact, a concrete function is defined by:
    . shape and dtype if a *tensor* is passed
    . id() for a primitive python type
It seems that numpy arrays are automatically casted to tensors. Thus, when passing an image to this function, 
we can check the generated concrete function;
    g = energy.get_concrete_function(array, init_level_set, lambda_1, lambda_2).graph
    g.inputs
has been traced for inputs corresponding to:
    [<tf.Tensor 'image:0' shape=(5, 5) dtype=float32>, <tf.Tensor 'phi:0' shape=(5, 5) dtype=float32>, 
    <tf.Tensor 'lambda_1:0' shape=(1,) dtype=float32>, <tf.Tensor 'lambda_2:0' shape=(1,) dtype=float32>]
Thus, if we call again the function with a different value for our parameters, it won't be traced. However, 
if we call it for another array shape, then it *will* be traced again. 
Interestingly, if we define:
    lambda_1 = np.array(1, dtype='float32')
Then it will be traced for *every* different value of lambda_1. Instead, choosing:
    lambda_1 = np.array([1], dtype='float32')
Seems to enable an automatic cast to tensor, and thus will not be traced for every value.
"""


@tf.function
def curvature(phi, return_normalization=False):
    """
Compute the curvature of a function. Discretization used by Chan and Vese, from [1].
    Parameters
    ----------
    phi: a 2-dim tensor.
    return_normalization: bool. If True, returns the normalization constants too.

    Returns
    -------
    If return_normalization: a tuple of (curvature, normalization_constant). Otherwise, just the curvature.

[1] Nonlinear total variation based noise removal algorithms, Leonid I. Rudin 1, Stanley Osher and Emad Fatemi
    """
    eta = tf.constant(1e-16)

    padded_phi = tf.pad(phi, [[1, 1], [1, 1]], mode="constant", constant_values=0)

    phixp = padded_phi[1:-1, 2:] - padded_phi[1:-1, 1:-1]
    phixn = padded_phi[1:-1, 1:-1] - padded_phi[1:-1, :-2]
    phix0 = (padded_phi[1:-1, 2:] - padded_phi[1:-1, :-2]) / 2.0

    phiyp = padded_phi[2:, 1:-1] - padded_phi[1:-1, 1:-1]
    phiyn = padded_phi[1:-1, 1:-1] - padded_phi[:-2, 1:-1]
    phiy0 = (padded_phi[2:, 1:-1] - padded_phi[:-2, 1:-1]) / 2.0

    C1 = 1. / tf.sqrt(eta + phixp**2 + phiy0**2)
    C2 = 1. / tf.sqrt(eta + phixn**2 + phiy0**2)
    C3 = 1. / tf.sqrt(eta + phix0**2 + phiyp**2)
    C4 = 1. / tf.sqrt(eta + phix0**2 + phiyn**2)

    result = (padded_phi[1:-1, 2:] * C1 + padded_phi[1:-1, :-2] * C2 +
              padded_phi[2:, 1:-1] * C3 + padded_phi[:-2, 1:-1] * C4)

    if return_normalization:
        return result, (C1 + C2 + C3 + C4)
    else:
        return result


@tf.function
def dirac(phi, epsilon=1e-1):
    """
A discretized dirac function.

    Parameters
    ----------
    phi: the array on which to apply the dirac function.
    epsilon: the bandwidth parameter.

    Returns
    -------
    Array phi on which the dirac was applied.
    """
    return epsilon / (np.pi * (tf.pow(epsilon, 2) + tf.pow(phi, 2)))


# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None]), tf.TensorSpec(shape=[None, None]),
#                               tf.TensorSpec(shape=[1]), tf.TensorSpec(shape=[1])])
@tf.function
def energy_derivative(phi, image, dt, lambda_1, lambda_2, mu, epsilon=1e-1):
    print('Tracing energy_derivate...')
    mean_inside = tf.reduce_mean(tf.gather_nd(image, tf.where(phi > 0)))
    mean_outside = tf.reduce_mean(tf.gather_nd(image, tf.where(phi <= 0)))
    phi_curvature, curv_normalization = curvature(phi, return_normalization=True)
    derivative = (
                - lambda_1 * tf.pow(image - mean_inside, 2) + lambda_2 * tf.pow(image - mean_outside, 2)
                + mu * phi_curvature
               )

    dirac_phi = dirac(phi, epsilon=epsilon)
    new_phi = phi + dt * dirac_phi * derivative
    return new_phi / (1 + dt * dirac_phi * mu * curv_normalization)


# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None]), tf.TensorSpec(shape=[None, None]),
#                               tf.TensorSpec(shape=[1]), tf.TensorSpec(shape=[1])])
@tf.function
def energy(image, phi, lambda_1, lambda_2):
    print("Tracing energy function...")
    mean_inside = tf.reduce_mean(tf.gather_nd(image, tf.where(phi > 0)))
    mean_outside = tf.reduce_mean(tf.gather_nd(image, tf.where(phi <= 0)))
    result = (lambda_1 * tf.reduce_sum(tf.pow(tf.gather(image, tf.where(phi > 0)) - mean_inside, 2))
              + lambda_2 * tf.reduce_sum(tf.pow(tf.gather(image, tf.where(phi <= 0)) - mean_outside, 2)))
    return result


def tensor_chan_vese(image, n_iter, dt, lambda_1=1., lambda_2=1., mu=.1, init_type='checkerboard',
                     extended_output=False, print_rate=100):
    """
A tensorflow implementation of a basic chan vese segmentation [1].
Designed as a proof of concept to see whether graph and gpu use can be useful for these type of algorithms.

[1] "Active Contours Without Edges", Tony F. Chan and Luminita A. Vese

    Parameters
    ----------
    image : np.ndarray
        2-dim numpy array. Image to segment.
    n_iter : int
        The number of steps for the flow
    dt : float
        The time parameter
    lambda_1 : float
        The constraint on the inside. The higher, the more homogeneous the interior gets.
    lambda_2 : float
        The constraint on the outside.
    mu :
        The regularization parameter. Suggested: between 0 and .25.
    init_type : str
        Initialization of the level-sets. Method taken from scikit-image package. Pass it either a string or a numpy
        array whose shape is the same of the image shape.
    extended_output : bool
        If False, returns the final level-set. Otherwise, returns final level-set, list of energies.

    Returns : np.ndarray | np.ndarray, list
    -------
        if extended_output:
            final level-set, list of energies
        else:
            final level-set

Overall, it's just an example of how tf.function can be used. It may actually be useful for very large image where
device placement can come in handy.

Note that there is no clear way in the literature on how to handle specifically the regularization parameter. Here,
we used the implementation from sklearn, which normalizes by every value. In the end, the user should try different
values and see what works best.
    """
    image = (image - image.min()) / (image.max() - image.min())
    phi = _cv_init_level_set(init_type, image.shape)

    image = image.astype('float32')
    phi = phi.astype('float32')
    lambda_1 = np.array([lambda_1], dtype='float32')
    lambda_2 = np.array([lambda_2], dtype='float32')

    if extended_output:
        energies = []

    time_all = time()
    time_batch = time()
    for i in range(n_iter):
        phi = energy_derivative(phi=phi, image=image, dt=dt, lambda_1=lambda_1, lambda_2=lambda_2, mu=mu)
        if extended_output:
            energies.append(energy(image, phi, lambda_1, lambda_2))
        if i % print_rate == 0 and i != 0:
            print(f'[{i:3d}] Did 100 in {time()-time_batch:.2f}s.')
            time_batch = time()
    print(f'[END] Did {n_iter} in {time() - time_all:.2f}s.')
    if extended_output:
        return phi, energies
    else:
        return phi


if __name__ == '__main__':
    array = np.load('resources/191ir.npy')[:500, :500]
    # curvature(array)


    t1 = time()
    phi_tensor = tensor_chan_vese(array, 1000, .5, lambda_1=1, lambda_2=1, mu=0.1, init_type='checkerboard')
    print(f"Did Tensor Chan Vese in {time() - t1:.1f}s")

    from skimage.segmentation import chan_vese
    t1 = time()
    phi_classic = chan_vese(array, max_iter=1000, tol=0, init_level_set='checkerboard')
    print(f"Did standard Chan Vese in {time() - t1:.1f}s")

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(phi_tensor, cmap='coolwarm')
    axes[0].set_title('Level-set of tensor')
    axes[1].imshow(array, cmap='gray')
    # axes[1].contour(phi_classic, [.5], colors='r')
    axes[1].contour(phi_tensor, [.5], colors='g')
    axes[1].contour(phi_classic, [.5], colors='r')
    axes[1].set_title('Raw image with contour')
    plt.show()
