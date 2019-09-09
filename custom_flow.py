"""
Here, the aim is to compute a flow which has a criterium close to the one showed in biorules_segmentation.py.
More specifically:
    * We have a multi channel image
    * We have some rules of type A => B or A => not B
    * So we minimize an energy such as :
        $ int_\Omega |I_A| |I_A - I_B| - |I_C - I_D| H(phi > 0) dx$
                      _______^_______    _____^_____
                           A => B        C => not D
We will take the interior as the positive part of the level-set.

----

The energy is not properly defined. We can't use a levelset formulation as is. We must take into account the whole
pictures. Otherwise, the level-set will only expand OR shrink continuously.
"""
import numpy as np
import skimage.draw as draw
from skimage.segmentation._chan_vese import _cv_init_level_set
import matplotlib.pyplot as plt


def dirac(x, eps=1e-1):
    return eps / (np.pi * (np.power(eps, 2) + np.power(x, 2)))


def compute_energy(image, phi):
    interior = image[phi > 0]
    mean_interior = np.mean(interior, axis=0)
    mean_interior = np.flip(mean_interior)
    score = np.power(interior - mean_interior, 2)
    score = np.mean(score, axis=0)
    return np.mean(score)


def compute_derivative(image: np.ndarray, phi):
    n_rows, n_cols, n_features = image.shape
    interior = image[phi > 0]
    image = image.reshape((n_rows * n_cols, 2))
    mean_interior = np.mean(interior, axis=0)
    mean_interior = np.flip(mean_interior)
    shift = np.power(image - mean_interior, 2)
    shift = np.reshape(shift, (n_rows, n_cols, n_features))
    shift = np.mean(shift, axis=2)
    shift = dirac(phi) * shift
    return shift


target = draw.circle(32, 32, 16)
IMAGE = np.zeros((64, 64, 2), dtype=np.float32)
IMAGE[..., 1] = 1
IMAGE[..., 0][target] = 1
IMAGE[..., 1][target] = 0

phi = _cv_init_level_set("disk", IMAGE.shape[:-1])
derivative = compute_derivative(IMAGE, phi)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(IMAGE[..., 0])
axes[0].contour(phi, [0], colors='r')
axes[1].imshow(IMAGE[..., 1])
axes[1].contour(phi, [0], colors='r')
im = axes[2].imshow(derivative, cmap='coolwarm')
plt.colorbar(im, ax=axes[2])
print(compute_energy(IMAGE, phi))
