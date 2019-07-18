from time import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation._chan_vese import _cv_init_level_set

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
# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None]), tf.TensorSpec(shape=[None, None]),
#                               tf.TensorSpec(shape=[1]), tf.TensorSpec(shape=[1])])
@tf.function
def energy_derivative(image, phi, lambda_1, lambda_2, epsilon=1e-1):
    print('Tracing energy_derivate...')
    mean_inside = tf.reduce_mean(tf.gather(image, tf.where(phi > 0)))
    mean_outside = tf.reduce_mean(tf.gather(image, tf.where(phi <= 0)))
    derivate = - lambda_1 * tf.pow(image - mean_inside, 2) + lambda_2 * tf.pow(image - mean_outside, 2)
    #                                          dirac(phi(x))
    #                      ___________________________^______________________________
    derivate = derivate * (epsilon / (np.pi * (tf.pow(epsilon, 2) + tf.pow(phi, 2))))
    return derivate


# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None]), tf.TensorSpec(shape=[None, None]),
#                               tf.TensorSpec(shape=[1]), tf.TensorSpec(shape=[1])])
@tf.function
def energy(image, phi, lambda_1, lambda_2):
    print("Tracing energy function...")
    mean_inside = tf.reduce_mean(tf.gather(image, tf.where(phi > 0)))
    mean_outside = tf.reduce_mean(tf.gather(image, tf.where(phi <= 0)))
    result = (lambda_1 * tf.reduce_sum(tf.pow(tf.gather(image, tf.where(phi > 0)) - mean_inside, 2))
              + lambda_2 * tf.reduce_sum(tf.pow(tf.gather(image, tf.where(phi <= 0)) - mean_outside, 2)))
    return result


def tensor_chan_vese(image, n_iter, dt, lambda_1=1., lambda_2=1., init_type='checkerboard',
                     extended_output=False):
    """
A tensorflow implementation of a basic chan vese segmentation [1]. No smoothing term is included.
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
    init_type : str
        Initialization of the level-sets. Method taken from scikit-image package. Pass it either a string or a numpy
        array whose shape is the same of the image shape.
    extended_output : bool
        If False, returns the final level-set. Otherwise, returns final level-set, list of energies.
    Returns : np.ndarray | np.ndarray, list
        if extended_output:
            final level-set, list of energies
        else:
            final level-set
    -------
In theory, such implementation could be faster than standard python implementation. Yet, it appears that the overhead
introduced to build the graph makes the overall slower. The computations made here may not be complex enough to gain
some speed over numpy computations, which are implemented in C.

Overall, it's just an example of how tf.function can be used. It may actually be useful for very large image where
device placement can come in handy.
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
        derivative = energy_derivative(image, phi, lambda_1, lambda_2)
        phi += dt * derivative
        if extended_output:
            energies.append(energy(image, phi, lambda_1, lambda_2))
        if i % 100 == 0 and i != 0:
            print(f'[{i:3d}] Did 100 in {time()-time_batch:.2f}s.')
            time_batch = time()
    print(f'[END] Did {n_iter} in {time() - time_all:.2f}s.')
    if extended_output:
        return phi, energies
    else:
        return phi

if __name__ == '__main__':
    array = np.load('resources/191Ir.npy')[:100, :100]

    phi = tensor_chan_vese(array, 1000, .1, 1, 1, 'small disk')

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(array, cmap='gray')
    axes[0].set_title('Raw image')
    axes[1].imshow(phi, cmap='coolwarm')
    axes[1].set_title('Level sets')
    axes[2].imshow(array, cmap='gray')
    axes[2].contour(phi, [.5], colors='r')
    axes[2].set_title('Raw image with contour')
    plt.show()
