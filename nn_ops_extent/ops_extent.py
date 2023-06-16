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
    return tf.multiply(tf.transpose(a, perm=perm_order), expand_tensor_dims_recursive(b, 2, 1))


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
    return expand_tensor_dims_recursive(inputs, expand_iters - 1, axis)


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
                                                                            (fft_filters * tf.math.conj(
                                                                                fft_filters) + lambds ** 2)),
                                                                perm_order=(0, 2, 1))))
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
    :param filters: the filter to be deconvolved of shape (height, width, #filters) - h & w must match with input_mat
    :param lambds: The SNR of the input image of shape (#filters, 1)
    :return: The deconvolved image.
    """

    # Transpose input matrices for propper form to apply FFT2D
    input_mat = tf.transpose(input_mat, perm=[0, 3, 1, 2])

    # Complete possibly real input vector with zeros for imaginary parts
    input_mat = real_to_complex_tensor(input_mat)
    filters = real_to_complex_tensor(filters)

    # Generate the FFT of filter's transfer function and input matrixes
    fft_filters = tf.signal.fft2d(filters)
    fft_input = tf.signal.fft2d(input_mat)

    # Compute simple Wiener deconvolution method 1 kinda deprecated
    # Match SNRs & filters shape
    # lambds = real_to_complex_tensor(lambds)
    # lambds = tf.broadcast_to(lambds[:, None], (tf.shape(lambds)[0], tf.shape(input_mat)[1], tf.shape(input_mat)[2]))
    # deconvolved = tf.math.real(tf.signal.ifft2d(outer_elementwise(fft_input, (tf.math.conj(fft_filters) /
    #                                                                          (fft_filters * tf.math.conj(fft_filters) +
    #                                                                           lambds ** 2)), perm_order=(0, 1, 2, 3))))
    # deconvolved = tf.reshape(deconvolved, (tf.shape(deconvolved)[1],
    #                                                    tf.shape(deconvolved)[0] * tf.shape(deconvolved)[2],
    #                                                    tf.shape(deconvolved)[3],
    #                                                    tf.shape(deconvolved)[4]))

    input_snr = tf.reduce_mean(tf.abs(fft_input) ** 2) / lambds

    g_right_hand = (1 / (1 + 1 / ((tf.abs(fft_filters) ** 2) * input_snr)))
    g_right_hand = tf.cast(g_right_hand, tf.complex64)

    g_freq_domain = (1 / fft_filters) * g_right_hand

    x_est_freq_domain = outer_elementwise(fft_input, g_freq_domain, perm_order=(0, 1, 2, 3))
    x_est_freq_domain = tf.reshape(x_est_freq_domain, (tf.shape(x_est_freq_domain)[1],
                                                       tf.shape(x_est_freq_domain)[0] * tf.shape(x_est_freq_domain)[2],
                                                       tf.shape(x_est_freq_domain)[3],
                                                       tf.shape(x_est_freq_domain)[4]))

    deconvolved = tf.math.real(tf.signal.ifft2d(x_est_freq_domain))

    # Make back to conventional shape of (batch, height, width, channels)
    deconvolved = tf.transpose(deconvolved, perm=(0, 2, 3, 1))

    return deconvolved


@tf.function
def denoise_tv_chambolle_nd(image, weights, max_num_iter=200):
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    :param image : ndarray
        n-D input data to be denoised.
    :param weights: 1D tensor with the weights for each channel
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    :param max_num_iter : int
        Maximal number of iterations used for the optimization.
    Returns
    -------
    out : ndarray
        Denoised array of floats.
    Notes
    -----
    Rudin, Osher and Fatemi algorithm.

    """

    assert weights.shape[0] == image.shape[-1] or weights.shape[0] == 1, \
        'Weights must have same size 1 or equal with number of image channels'

    ndim = image.get_shape().ndims
    weights = tf.cast(weights, image.dtype)

    p = tf.zeros((ndim,) + image.shape, dtype=image.dtype)
    out = tf.zeros_like(image)

    slice_d_ax0 = [slice(1, None, None), slice(None, None, None), slice(None, None, None)]
    conct_d_ax0 = [slice(None, 1, None), slice(None, None, None), slice(None, None, None)]
    slice_p_ax0 = [0, slice(0, -1, None), slice(None, None, None), slice(None, None, None)]

    slice_d_ax1 = [slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    conct_d_ax1 = [slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
    slice_p_ax1 = [1, slice(None, None, None), slice(0, -1, None), slice(None, None, None)]

    slice_d_ax2 = [slice(None, None, None), slice(None, None, None), slice(1, None, None)]
    conct_d_ax2 = [slice(None, None, None), slice(None, None, None), slice(None, 1, None)]
    slice_p_ax2 = [2, slice(None, None, None), slice(None, None, None), slice(0, -1, None)]

    for i in tf.range(max_num_iter):
        if i > 0:
            # d will be the (negative) divergence of p
            d = tf.reduce_sum(-p, 0)

            d_p_ax0 = d[slice_d_ax0] + p[slice_p_ax0]
            d = tf.concat([d[conct_d_ax0], d_p_ax0], axis=0)

            d_p_ax1 = d[slice_d_ax1] + p[slice_p_ax1]
            d = tf.concat([d[conct_d_ax1], d_p_ax1], axis=1)

            d_p_ax2 = d[slice_d_ax2] + p[slice_p_ax2]
            d = tf.concat([d[conct_d_ax2], d_p_ax2], axis=2)

            out = image + d * tf.cast((i != 0), d.dtype)

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        diff_g_ax0 = tf.experimental.numpy.diff(out, axis=0)
        diff_g_ax0 = tf.pad(diff_g_ax0, [[0, 1], [0, 0], [0, 0]])

        diff_g_ax1 = tf.experimental.numpy.diff(out, axis=1)
        diff_g_ax1 = tf.pad(diff_g_ax1, [[0, 0], [0, 1], [0, 0]])

        diff_g_ax2 = tf.experimental.numpy.diff(out, axis=2)
        diff_g_ax2 = tf.pad(diff_g_ax2, [[0, 0], [0, 0], [0, 1]])

        g = tf.concat([[diff_g_ax0], [diff_g_ax1], [diff_g_ax2]], axis=0)

        norm = tf.sqrt(tf.reduce_sum(g ** 2, axis=0))[None, ...]
        tau = 1. / (2. * ndim)
        norm *= tau / weights
        norm += 1.
        p = p - tau * g
        p = p / norm

    return out
