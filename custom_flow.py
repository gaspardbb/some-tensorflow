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
from scipy.ndimage import distance_transform_edt
from skimage.draw import circle
from skimage.segmentation._chan_vese import _cv_init_level_set
import matplotlib.pyplot as plt
from skimage.util import noise
from tqdm import trange


def dirac(x, eps=1e-1):
    return eps / (np.pi * (np.power(eps, 2) + np.power(x, 2)))


def compute_energy(image, phi):
    interior = image[phi > 0]
    mean_interior = np.mean(interior, axis=0)
    mean_interior = np.flip(mean_interior)

    score_interior = np.power(interior - mean_interior, 2)
    score_interior = np.mean(score_interior, axis=0)

    exterior = image[phi <= 0]
    mean_exterior = np.mean(exterior, axis=0)
    mean_exterior = np.flip(mean_exterior)

    score_exterior = np.power(exterior - mean_exterior, 2)
    score_exterior = np.mean(score_exterior, axis=0)

    return np.mean(score_interior) - np.mean(score_exterior)


def compute_derivative(image: np.ndarray, phi):
    n_rows, n_cols, n_features = image.shape

    interior = image[phi > 0]
    exterior = image[phi <= 0]

    image = image.reshape((n_rows * n_cols, 2))

    mean_interior = np.mean(interior, axis=0)
    mean_interior = np.flip(mean_interior)

    mean_exterior = np.mean(exterior, axis=0)
    mean_exterior = np.flip(mean_exterior)

    shift_interior = np.power(image - mean_interior, 2)
    shift_interior = np.reshape(shift_interior, (n_rows, n_cols, n_features))
    shift_interior = np.mean(shift_interior, axis=2)

    shift_exterior = np.power(image - mean_exterior, 2)
    shift_exterior = np.reshape(shift_exterior, (n_rows, n_cols, n_features))
    shift_exterior = np.mean(shift_exterior, axis=2)

    shift = dirac(phi) * (shift_interior - shift_exterior)
    return shift


def apply_derivative(image: np.ndarray, phi: np.ndarray, dt: float):
    n_rows, n_cols, n_features = image.shape

    interior = image[phi > 0]
    exterior = image[phi <= 0]

    image = image.reshape((n_rows * n_cols, 2))

    mean_interior = np.mean(interior, axis=0)
    mean_interior = np.flip(mean_interior)

    mean_exterior = np.mean(exterior, axis=0)
    mean_exterior = np.flip(mean_exterior)

    shift_interior = np.power(image - mean_interior, 2)
    shift_interior = np.reshape(shift_interior, (n_rows, n_cols, n_features))
    shift_interior = np.mean(shift_interior, axis=2)

    shift_exterior = np.power(image - mean_exterior, 2)
    shift_exterior = np.reshape(shift_exterior, (n_rows, n_cols, n_features))
    shift_exterior = np.mean(shift_exterior, axis=2)

    dirac_phi = dirac(phi)
    shift = dirac_phi * (shift_interior - shift_exterior)

    new_phi = phi + dt*shift

    return new_phi / (1 + dt * dirac_phi)


def signed_distance(phi: np.ndarray, threshold: float = 0):
    """
Return the signed distance from the `threshold` level-set. Parts above threshold are counted positively.
    :param phi:
    :param threshold:
    :return:
    """
    return distance_transform_edt(phi > threshold) - distance_transform_edt(phi < threshold)


def circle_initialization(r: float or list, c: float or list, radius: float or list, shape):
    if type(r) is not list:
        r = [r]
    if type(c) is not list:
        c = [c]
    if type(radius) is not list:
        radius = [radius]
    assert len(r) == len(c) == len(radius)

    output = np.ones(shape)
    for (r_, c_, radius_) in [(r[i], c[i], radius[i]) for i in range(len(r))]:
        output[circle(r_, c_, radius_)] = -1
    return signed_distance(output)


# target = draw.circle(32, 32, 16)
# IMAGE = np.zeros((64, 64, 2), dtype=np.float32)
# IMAGE[..., 1] = 1
# IMAGE[..., 0][target] = 1
# IMAGE[..., 1][target] = 0
#
# IMAGE[..., 0] = noise.random_noise(IMAGE[..., 0])
# IMAGE[..., 1] = noise.random_noise(IMAGE[..., 1])

membrane = np.load('resources/158gd.npy')
nuclei = np.load('resources/191ir.npy')
membrane = membrane[200:300, 200:300]
nuclei = nuclei[200:300, 200:300]

IMAGE = np.stack([nuclei, membrane], axis=2)

dt = .5

phi = _cv_init_level_set("small disk", IMAGE.shape[:-1])
# phi = - circle_initialization(32, 24, 16, IMAGE.shape[:-1])

phi_init = phi.copy()
first_derivative = compute_derivative(IMAGE, phi)
first_score = compute_energy(IMAGE, phi_init)

for i in trange(10000):
    phi = apply_derivative(IMAGE, phi, dt)

last_derivative = compute_derivative(IMAGE, phi)
last_score = compute_energy(IMAGE, phi)


fig, axes = plt.subplots(3, 2)
axes[0, 0].imshow(IMAGE[..., 0])
axes[0, 0].contour(phi_init, [0], colors='r')
axes[0, 0].set_title("Channel 0")
axes[0, 1].imshow(IMAGE[..., 1])
axes[0, 1].contour(phi_init, [0], colors='r')
axes[0, 1].contour(phi, [0], colors='g')
axes[0, 1].set_title("Channel 1")

im = axes[1, 0].imshow(first_derivative, cmap='coolwarm')
axes[1, 0].set_title("First derivative")
plt.colorbar(im, ax=axes[1, 0])

im2 = axes[1, 1].imshow(phi_init, cmap='coolwarm')
plt.colorbar(im2, ax=axes[1, 1])
axes[1, 1].set_title("First level-set (%4.3f)" % first_score)

im3 = axes[2, 0].imshow(last_derivative, cmap='coolwarm')
axes[2, 0].set_title("Last derivative")
plt.colorbar(im3, ax=axes[2, 0])

im4 = axes[2, 1].imshow(phi, cmap='coolwarm')
plt.colorbar(im4, ax=axes[2, 1])
axes[2, 1].set_title("Last level-set (%4.3f)" % last_score)

print(compute_energy(IMAGE, phi))