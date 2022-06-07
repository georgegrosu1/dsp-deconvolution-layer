import numpy as np
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer


class Deconvolution(Layer):

    def __int__(self,
                filters: int = None,
                kernel_size=None,
                padding=None,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                lambd_initializer='random_normal',
                bias_initializer='zeros',
                kernel_regularizer=None,
                lambd_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                conv_op=None,
                **kwargs):
        super(Deconvolution, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.lambd_initializer = initializers.get(lambd_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.lambd_regularizer = regularizers.get(lambd_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = activity_regularizer
        self.use_bias = use_bias
        self.w = None
        self.s = None
        self.b = None

    def _pad_input(self, input_tensor):
        assert isinstance(self.padding, tuple) and len(self.padding) <= 3, 'Padding is specified by tuple with left, ' \
                                                                           'right padding lengths and mode (optionally)'
        if len(self.padding) == 3:
            return np.pad(input_tensor, (self.padding[0], self.padding[1]), mode=self.padding[-1])
        else:
            return np.pad(input_tensor, (self.padding[0], self.padding[1]))

    def _match_filters_to_input_padding(self, input_shape):
        len_pad = input_shape[-1] - self.w.shape[-1]
        return tf.Variable(np.pad(self.w,
                                  (0, len_pad),
                                  'constant')[:-len_pad],
                           trainable=True,
                           dtype='float32')

    def build(self, input_shape):
        self.w = self.add_weight(shape=(self.filters, self.kernel_size[0]),
                                 initializer=self.kernel_initializer,
                                 trainable=True,
                                 dtype='float32')
        if self.w.shape[-1] < input_shape:
            self.w = self._match_filters_to_input_padding(input_shape)
        self.s = self.add_weight(shape=(self.filters, 1),
                                 initializer=self.lambd_initializer,
                                 trainable=True,
                                 dtype='float32')
        if self.use_bias:
            self.b = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     trainable=True,
                                     dtype='float32')

    def call(self, inputs, training=None):
        x = None
        if self.padding is not None:
            x = self._pad_input(inputs)
        if self.use_bias:
            x = x + self.b
        x = self.activation(x)
        return x

    def _serialize_to_tensors(self):
        pass

    def _restore_from_tensors(self, restored_tensors):
        pass
