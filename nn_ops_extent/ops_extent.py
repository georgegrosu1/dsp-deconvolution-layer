import tensorflow as tf
from tensorflow.python.keras import backend as bend


@tf.function
def element_wise_oriented(a, b):
    outs = tf.TensorArray(tf.complex64, size=0, dynamic_size=True)
    # Iterate over input vectors in a and compute element-wise multiplication with b for each vector
    for i in tf.range(a[0].shape[0]):
        outs = outs.write(i, a[0][i] * b)
    return outs.stack()


@tf.function
def expand_tensor_dims_recursive(inputs, expand_iters=1, axis=0):
    if expand_iters == 1:
        return bend.expand_dims(inputs, axis)
    inputs = bend.expand_dims(inputs, axis)
    return expand_tensor_dims_recursive(inputs, expand_iters-1, axis)


@tf.function
def real_to_complex_tensor(input_tensor):
    if input_tensor.dtype != tf.complex64:
        return tf.cast(input_tensor, tf.complex64)
    return input_tensor


@tf.function
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
    input_vect = real_to_complex_tensor(input_vect)
    lambds = real_to_complex_tensor(lambds)
    # Generate the Fourier transforms of the input vectors and filter's transfer function
    fft_input = tf.signal.fft(input_vect)
    fft_filters = tf.signal.fft(tf.complex(filters[0], filters[-1]))
    # Compute simple Wiener deconvolution
    deconvolved = tf.math.real(tf.signal.ifft(element_wise_oriented(fft_input, (tf.math.conj(fft_filters) /
                                              (fft_filters * tf.math.conj(fft_filters) + lambds**2)))))

    return deconvolved
