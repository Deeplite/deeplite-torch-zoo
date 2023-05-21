# Code modified from: https://github.com/hendrycks/robustness

import ctypes
import pathlib
from io import BytesIO

import cv2
import numpy as np
import skimage as sk
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

from deeplite_torch_zoo.utils.registry import Registry

DISTORTION_REGISTRY = Registry()


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,
)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    import math

    mapsize = 2 ** math.ceil(math.log2(mapsize))
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[0], img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    chw = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + chw], (zoom_factor, zoom_factor, 1), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top : trim_top + h, trim_top : trim_top + w]


@DISTORTION_REGISTRY.register('gaussian_noise')
def gaussian_noise(x, severity=1):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


@DISTORTION_REGISTRY.register('shot_noise')
def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


@DISTORTION_REGISTRY.register('impulse_noise')
def impulse_noise(x, severity=1):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255.0, mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


@DISTORTION_REGISTRY.register('speckle_noise')
def speckle_noise(x, severity=1):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


@DISTORTION_REGISTRY.register('gaussian_blur')
def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


@DISTORTION_REGISTRY.register('glass_blur')
def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) * 255)

    h_image, w_image = x.shape[0], x.shape[1]

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(h_image - c[1], c[1], -1):
            for w in range(w_image - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255


@DISTORTION_REGISTRY.register('defocus_blur')
def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


@DISTORTION_REGISTRY.register('motion_blur')
def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    w, h = x.shape[0], x.shape[1]
    x = PILImage.fromarray(x)
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (w, h):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


@DISTORTION_REGISTRY.register('zoom_blur')
def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


@DISTORTION_REGISTRY.register('fog')
def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    w, h = x.shape[0], x.shape[1]
    x = np.array(x) / 255.0
    max_val = x.max()
    x += (
        c[0]
        * plasma_fractal(mapsize=max(w, h), wibbledecay=c[1])[:w, :h][..., np.newaxis]
    )
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


@DISTORTION_REGISTRY.register('frost')
def frost(x, severity=1):
    HERE = pathlib.Path(__file__).parent
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    w, h = x.shape[0], x.shape[1]
    idx = np.random.randint(5)
    filename = [
        HERE / './frost_imgs/frost1.png',
        HERE / './frost_imgs/frost2.png',
        HERE / './frost_imgs/frost3.png',
        HERE / './frost_imgs/frost4.jpg',
        HERE / './frost_imgs/frost5.jpg',
        HERE / './frost_imgs/frost6.jpg',
    ][idx]
    frost = cv2.imread(str(filename))

    # randomly crop and convert to rgb
    frost = cv2.resize(frost, (h, w))
    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


@DISTORTION_REGISTRY.register('snow')
def snow(x, severity=1):
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(
        size=x.shape[:2], loc=c[0], scale=c[1]
    )  # [:2] for monochrome
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray(
        (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L'
    )
    output = BytesIO()

    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())
    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = (
        cv2.imdecode(
            np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        / 255.0
    )
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(
        x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[..., np.newaxis] * 1.5 + 0.5
    )
    snow_layer = cv2.resize(snow_layer, (x.shape[1], x.shape[0]))[..., np.newaxis]
    snow_layer_rot = cv2.resize(np.rot90(snow_layer, k=2), (x.shape[1], x.shape[0]))[
        ..., np.newaxis
    ]
    return np.clip(x + snow_layer + snow_layer_rot, 0, 1) * 255


@DISTORTION_REGISTRY.register('spatter')
def spatter(x, severity=1):
    c = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1),
    ][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate(
            (
                175 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate(
            (
                63 / 255.0 * np.ones_like(x[..., :1]),
                42 / 255.0 * np.ones_like(x[..., :1]),
                20 / 255.0 * np.ones_like(x[..., :1]),
            ),
            axis=2,
        )

        color *= m[..., np.newaxis]
        x *= 1 - m[..., np.newaxis]

        return np.clip(x + color, 0, 1) * 255


@DISTORTION_REGISTRY.register('contrast')
def contrast(x, severity=1):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


@DISTORTION_REGISTRY.register('brightness')
def brightness(x, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


@DISTORTION_REGISTRY.register('saturate')
def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    # x = np.transpose(x, (2, 0, 1))
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


@DISTORTION_REGISTRY.register('jpeg_compression')
def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    w, h = x.shape[0], x.shape[1]
    x = PILImage.fromarray(x)
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return np.array(x)


@DISTORTION_REGISTRY.register('pixelate')
def pixelate(x, severity=1):
    w, h = x.shape[0], x.shape[1]
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = cv2.resize(x, (int(w * c), int(h * c)))
    x = cv2.resize(x, (w, h))

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
@DISTORTION_REGISTRY.register('elastic_transform')
def elastic_transform(image, severity=1):
    c = [
        (
            244 * 2,
            244 * 0.7,
            244 * 0.1,
        ),  # 244 should have been 224, but ultimately nothing is incorrect
        (244 * 2, 244 * 0.08, 244 * 0.2),
        (244 * 0.05, 244 * 0.01, 244 * 0.02),
        (244 * 0.07, 244 * 0.01, 244 * 0.02),
        (244 * 0.12, 244 * 0.01, 244 * 0.02),
    ][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode='reflect').reshape(shape),
            0,
            1,
        )
        * 255
    )


@DISTORTION_REGISTRY.register('barrel')
def barrel(x, severity=1):
    c = [
        (0, 0.03, 0.03),
        (0.05, 0.05, 0.05),
        (0.1, 0.1, 0.1),
        (0.2, 0.2, 0.2),
        (0.1, 0.3, 0.6),
    ][severity - 1]

    output = BytesIO()
    PILImage.fromarray(x).save(output, format='PNG')

    x = WandImage(blob=output.getvalue())
    x.distort('barrel', c)

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
