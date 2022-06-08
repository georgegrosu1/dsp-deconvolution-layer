import tensorflow as tf


def deconv1d(input_vect, filters, lambds):
    """
    We take the Fourier transform of the input vector and the filter's transfer function, multiply them together, divide by
    the sum of the squared filter's transfer function and the regularization parameter, and then take the inverse Fourier
    transform of the result
    
    :param input_vect: The input vector to be deconvolved
    :param filters: the filter's transfer function
    :param lambds: SNR values
    :return: The deconvolved signal.
    """
    # Complete possibly real input vector with zeros for imaginary parts
    if all(tf.math.imag(input_vect)):
        input_vect = tf.complex(input_vect, tf.zeros_like(input_vect, dtype='float32'))
    # Generate the Fourier transforms of the input vectors and filter's transfer function
    fft_input = tf.constant(tf.signal.fft(input_vect))
    fft_filters = tf.constant(tf.signal.fft(tf.complex(filters[0], filters[-1])))
    # Compute simple Wiener deconvolution
    deconvolved = tf.constant(tf.math.real(tf.signal.ifft(tf.divide(tf.multiply(fft_input, tf.math.conj(fft_filters)),
                                                                    (tf.add(tf.multiply(fft_filters,
                                                                                        tf.math.conj(fft_filters)),
                                                                            tf.pow(lambds, 2)))))))

    return deconvolved
