import copy

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from nn_ops_extent.ops_extent import deconv1d


class Deconvolution(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='lecun_normal',
                 lambd_initializer='lecun_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 lambd_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Deconvolution, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.lambd_initializer = initializers.get(lambd_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.lambd_regularizer = regularizers.get(lambd_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.use_bias = use_bias
        self.w_real = None
        self.w_imag = None
        self.s = None
        self.b = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.w_real = self.add_weight(shape=(self.filters, self.kernel_size[0]),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      dtype='float32')
        self.w_imag = self.add_weight(shape=self.w_real.shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      dtype='float32')
        self._pad_filters2match_input(input_shape)
        self.s = self.add_weight(shape=(self.filters, 1),
                                 initializer=self.lambd_initializer,
                                 regularizer=self.lambd_regularizer,
                                 trainable=True,
                                 dtype='float32')
        if self.use_bias:
            self.b = self.add_weight(shape=(self.filters, 1),
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     trainable=True,
                                     dtype='float32')

    def call(self, inputs, *args, **kwargs):
        x = copy.copy(inputs)
        # Apply padding to input if specified
        if self.padding is not None:
            x = self._pad_input(x)
        # Apply Wiener deconvolution on inputs
        assert x.shape[-1] == self.w_real.shape[-1], 'Input and kernels must have equal shapes. Reduce filters ' \
                                                     'length, use input padding or increase input length.'
        x = deconv1d(input_vect=x, filters=(self.w_real, self.w_imag), lambds=self.s)
        # Apply bias accordingly
        if self.use_bias:
            x = x + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _pad_input(self, input_tensor):
        assert isinstance(self.padding, tuple) and len(self.padding) <= 2, 'Padding is specified by tuple with pad ' \
                                                                           'length and mode (optionally)'
        if len(self.padding) == 2:
            return tf.pad(input_tensor, ((0, 0), (0, self.padding[0])), mode=self.padding[-1])
        return tf.pad(input_tensor, ((0, 0), (0, self.padding[0])))

    def _pad_filters2match_input(self, input_shape):
        # Determine pad length
        if self.padding is not None:
            len_pad = input_shape[-1] + self.padding[0] - self.w_real.shape[-1]
        else:
            len_pad = input_shape[-1] - self.w_real.shape[-1]
        # Pad both real and imaginary weights
        if len_pad > 0:
            self.w_real = tf.pad(self.w_real, ((0, 0), (0, len_pad)), 'constant')
            self.w_imag = tf.pad(self.w_imag, ((0, 0), (0, len_pad)), 'constant')

    def _serialize_to_tensors(self):
        pass

    def _restore_from_tensors(self, restored_tensors):
        pass
