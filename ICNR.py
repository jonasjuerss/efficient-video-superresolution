import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer, GlorotUniform
from tensorflow.python.layers.utils import normalize_tuple

class ICNR(Initializer):
    """
    Inspired by: https://github.com/kostyaev/ICNR/blob/master/icnr.py
    """
    def __init__(self, scale: int, initializer=GlorotUniform()):
        """ICNR initializer for checkerboard artifact free transpose convolution
        Code adapted from https://github.com/kostyaev/ICNR
        Discussed at https://github.com/Lasagne/Lasagne/issues/862
        Original paper: https://arxiv.org/pdf/1707.02937.pdf
        Parameters
        ----------
        initializer : Initializer
            Initializer used for kernels (glorot uniform, etc.)
        scale : iterable of two integers, or a single integer
            Stride of the transpose convolution
            (a.k.a. scale factor of sub pixel convolution)
        """
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype):
        """

        Args:
            shape: (kernel_height, kernel_width, channels_in, scale * scale * channels_out (= 4 * 4 * 3))
            dtype:

        Returns:

        """
        if self.scale == 1:
            return self.initializer(shape)
        shape = list(shape)
        new_shape = shape[0:3] + [shape[3] // (self.scale ** 2)]
        # (kernel_height, kernel_width, channels_in, channels_out (=3))
        x = self.initializer(shape=new_shape, dtype=dtype)
        x = tf.repeat(x, repeats=self.scale * self.scale, axis=-1)
        #print(x)
        return x

# class ICNRSeparable(Initializer):
#     def __init__(self, scale: int, initializer=GlorotUniform()):
#         """ICNR initializer for checkerboard artifact free transpose convolution
#         Code adapted from https://github.com/kostyaev/ICNR
#         Discussed at https://github.com/Lasagne/Lasagne/issues/862
#         Original paper: https://arxiv.org/pdf/1707.02937.pdf
#         Parameters
#         ----------
#         initializer : Initializer
#             Initializer used for kernels (glorot uniform, etc.)
#         scale : iterable of two integers, or a single integer
#             Stride of the transpose convolution
#             (a.k.a. scale factor of sub pixel convolution)
#         """
#         self.scale = scale
#         self.initializer = initializer
#
#     def __call__(self, shape, dtype):
#         """
#
#         Args:
#             shape: (kernel_height, kernel_width, channels_in, scale * scale * channels_out (= 4 * 4 * 3))
#             dtype:
#
#         Returns:
#
#         """
#         x = self.initializer(shape=(shape[0], shape[1] // (self.scale ** 2)), dtype=dtype)
#         x = tf.repeat(x, repeats=self.scale * self.scale, axis=-1)
#         print(shape, x, x.shape)
#         return x