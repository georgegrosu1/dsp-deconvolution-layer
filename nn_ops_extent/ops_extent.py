import tensorflow as tf


@tf.function
def outer_elementwise(a, b):
    """
    It takes two tensors, `a` and `b`, and returns a tensor of the same shape as `a` where each element is the product
    of the corresponding element in `a` and the corresponding element in `b`, or the corresponding outer element-wise
    product
    :param a: (batch_size, timestamps, features)
    :param b: (features, len_features)
    """
    return tf.multiply(tf.transpose(a, perm=(0, 2, 1)), tf.expand_dims(tf.expand_dims(b, axis=1), axis=1))


@tf.function
def expand_tensor_dims_recursive(inputs, expand_iters=1, axis=0):
    """
    It takes a tensor and expands its dimensions by a specified number of times along a given axis
    :param inputs: the tensor to expand
    :param expand_iters: The number of times to expand the dimensions of the tensor, defaults to 1 (optional)
    :param axis: The axis to expand the dimensions of the tensor, defaults to 0 (optional)
    :return: A tensor with the same data as inputs but with a new dimension inserted at the specified position.
    """
    if expand_iters == 1:
        return tf.expand_dims(inputs, axis)
    inputs = tf.expand_dims(inputs, axis)
    return expand_tensor_dims_recursive(inputs, expand_iters-1, axis)


@tf.function
def real_to_complex_tensor(input_tensor):
    """
    It takes a tensor of real numbers and returns a tensor of complex numbers
    :param input_tensor: The tensor to be converted
    :return: The input tensor is being cast to a complex64 tensor.
    """
    if input_tensor.dtype != tf.complex64:
        return tf.cast(input_tensor, tf.complex64)
    return input_tensor


@tf.function
def deconv1d(input_vect, filters, lambds):
    """
    Take FFT of the input vector and the filter's transfer function, multiply them together, divide by the sum of the
    squared filter's transfer function and the regularization parameter, and then take the inverse Fourier
    transform of the result
    :param input_vect: The input vector to be deconvolved
    :param filters: the filter's transfer function
    :param lambds: SNR values
    :return: The deconvolved signal.
    """
    # Complete possibly real input vector with zeros for imaginary parts
    input_vect = real_to_complex_tensor(input_vect)
    lambds = real_to_complex_tensor(lambds)
    # Generate the FFT of filter's transfer function
    fft_filters = tf.signal.fft(tf.complex(filters[0], filters[-1]))
    # FFT all the signal batches of corresponding feature_idx
    fft_input = tf.signal.fft(input_vect)
    # Compute simple Wiener deconvolution
    deconvolved = tf.math.real(tf.signal.ifft(outer_elementwise(fft_input, (tf.math.conj(fft_filters) /
                                              (fft_filters * tf.math.conj(fft_filters) + lambds**2)))))
    # Reshape the resulted deconvoluted maps to normal shape of (batch, timestamps, features) where number of features
    # now is the product of initial features times the number of deconvolution filters (from each independent signal
    # results a set of deconvoluted signals equal to the number of filters)
    return tf.reshape(deconvolved, (tf.shape(input_vect)[0],
                                    tf.shape(input_vect)[1],
                                    tf.shape(input_vect)[-1] * tf.shape(filters[0])[0]))
