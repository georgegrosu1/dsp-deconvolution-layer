import tensorflow as tf


@tf.function
def outer_elementwise(a, b, perm_order):
    """
    It takes two tensors, `a` and `b`, and returns a tensor of the same shape as `a` where each element is the product
    of the corresponding element in `a` and the corresponding element in `b`, or the corresponding outer element-wise
    product
    :param perm_order: Tuple to indicate transposing order
    :param a: (batch_size, timestamps, features)
    :param b: (features, len_features)
    """
    return tf.multiply(tf.transpose(a, perm=perm_order), tf.expand_dims(tf.expand_dims(b, axis=1), axis=1))


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
                                              (fft_filters * tf.math.conj(fft_filters) + lambds**2)), perm=(0, 2, 1))))
    # Reshape the resulted deconvoluted maps to normal shape of (batch, timestamps, features) where number of features
    # now is the product of initial features times the number of deconvolution filters (from each independent signal
    # results a set of deconvoluted signals equal to the number of filters)
    return tf.reshape(deconvolved, (tf.shape(input_vect)[0],
                                    tf.shape(input_vect)[1],
                                    tf.shape(input_vect)[-1] * tf.shape(filters[0])[0]))


@tf.function
def deconv2d(input_mat, filters, lambds):
    """
    It takes the input matrix, the filters and the SNRs, and returns the deconvolved matrix
    :param input_mat: the input matrix to be deconvolved of shape (batch, width, height, channels)
    :param filters: the filter to be deconvolved of shape (width, height, #filters) - w & h must match with input_mat
    :param lambds: The SNR of the input image of shape (#filters, 1)
    :return: The deconvolved image.
    """
    # Store initial shape for later reconstruction to conventional shape
    init_shape = tf.shape(input_mat)

    # Match SNRs & filters shape
    lambds = tf.broadcast_to(lambds[:, None], (tf.shape(lambds)[0], tf.shape(input_mat)[1], tf.shape(input_mat)[2]))

    # Transpose input matrices for propper form to apply FFT2D
    input_mat = tf.transpose(input_mat, perm=[0, 3, 1, 2])
    filters = tf.transpose(filters, perm=[2, 0, 1])

    # Complete possibly real input vector with zeros for imaginary parts
    input_mat = real_to_complex_tensor(input_mat)
    filters = real_to_complex_tensor(filters)
    lambds = real_to_complex_tensor(lambds)

    # Generate the FFT of filter's transfer function and input matrixes
    fft_filters = tf.signal.fft2d(filters)
    fft_input = tf.signal.fft2d(input_mat)

    # Compute simple Wiener deconvolution
    deconvolved = tf.math.real(tf.signal.ifft(outer_elementwise(fft_input, (tf.math.conj(fft_filters) /
                                                                            (fft_filters * tf.math.conj(fft_filters) +
                                                                             lambds ** 2)), perm=(0, 1, 3, 2))))

    # Make back to conventional shape of (batch, width, height, channels)
    deconvolved = tf.transpose(deconvolved, perm=(1, 3, 4, 2, 0))
    deconvolved = tf.reshape(deconvolved, (init_shape[0],
                                           init_shape[1],
                                           init_shape[2],
                                           init_shape[-1] * tf.shape(filters)[0]))

    return deconvolved
