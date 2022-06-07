import numpy as np
import tensorflow as tf


def deconv1d(input_vect, filters, lambds):
    # Generate the Fourier transforms of the input vectors and filter's transfer function
    fft_input = tf.constant(np.fft.fft(input_vect))
    fft_filters = tf.constant(np.fft.fft(filters))
    # Compute simple Wiener deconvolution
    deconvolved = tf.constant(np.real(np.fft.ifft(tf.divide(tf.multiply(fft_input, np.conj(fft_filters)),
                                                            (tf.add(tf.multiply(fft_filters, np.conj(fft_filters)),
                                                                    lambds ** 2))))))

    return deconvolved
