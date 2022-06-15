import tensorflow as tf
from tensorflow.python.keras import backend as bend


@tf.function
def element_wise_oriented(a, b):
    outs = tf.TensorArray(tf.complex64, size=0, dynamic_size=True)
    # Iterate over input vectors in a and compute element-wise multiplication with b for each vector
    for batch_idx in tf.range(a.shape[0]):
        outs = outs.write(batch_idx, a[batch_idx] * b)
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
    # Generate the Fourier transforms of filter's transfer function
    fft_filters = tf.signal.fft(tf.complex(filters[0], filters[-1]))
    # Parse all the features of the batches, FFT them and perform deconvolution
    deconv_features = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for feature_idx in tf.range(bend.shape(input_vect)[-1]):
        # FFT all the signal batches of corresponding feature_idx
        fft_input = tf.signal.fft(input_vect[:, :, feature_idx])
        # Compute simple Wiener deconvolution
        deconvolved = tf.math.real(tf.signal.ifft(element_wise_oriented(fft_input, (tf.math.conj(fft_filters) /
                                                  (fft_filters * tf.math.conj(fft_filters) + lambds**2)))))
        # Reshape
        deconvolved = tf.reshape(deconvolved, (bend.shape(input_vect)[0],
                                               bend.shape(input_vect)[1],
                                               bend.shape(filters[0])[0]))
        deconv_features = deconv_features.write(feature_idx, deconvolved)
    deconv_features = deconv_features.stack()
    return tf.reshape(deconv_features, (bend.shape(input_vect)[0],
                                        bend.shape(input_vect)[1],
                                        bend.shape(input_vect)[-1] * bend.shape(filters[0])[0]))
