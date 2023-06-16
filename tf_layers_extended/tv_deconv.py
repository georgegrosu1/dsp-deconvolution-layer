import copy

import tensorflow as tf
from tensorflow.keras import backend as bend
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from nn_ops_extent import denoise_tv_chambolle_nd


class TVAbstractDeconvolution(tf.keras.layers.Layer):
    def __int__(self,
                max_iters,
                num_lambdas,
                activation,
                use_bias,
                lambdas_initializer,
                bias_initializer,
                lambdas_regularizer,
                bias_regularizer,
                activity_regularizer,
                trainable,
                name,
                **kwargs
                ):
        self.max_iters = max_iters
        self.num_lambdas = num_lambdas
        self.activation = activations.get(activation)
        self.lambdas_initializer = initializers.get(lambdas_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.lambdas_regularizer = regularizers.get(lambdas_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.use_bias = use_bias

        super(TVAbstractDeconvolution, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        def build(self, input_shape):
            raise NotImplementedError

        def call(self, inputs, *args, **kwargs):
            raise NotImplementedError

        def _serialize_to_tensors(self):
            pass

        def _restore_from_tensors(self, restored_tensors):
            pass

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'max_iters': self.max_iters,
                'num_lambdas': self.num_lambdas,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'lambdas_initializer': initializers.serialize(self.lambdas_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'lambdas_regularizer': regularizers.serialize(self.lambdas_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer)
            })
            return config


class TVDeconvolution2D(TVAbstractDeconvolution):

    def __init__(self,
                max_iters: int,
                num_lambdas: int,
                activation=None,
                use_bias=True,
                lambdas_initializer='glorot_normal',
                bias_initializer='zeros',
                lambdas_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                **kwargs):

        super(TVDeconvolution2D, self).__int__(
            max_iters=max_iters,
            num_lambdas=num_lambdas,
            activation=activation,
            use_bias=use_bias,
            lambdas_initializer=lambdas_initializer,
            bias_initializer=bias_initializer,
            lambdas_regularizer=lambdas_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        self.lambdas = None
        self.b = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.lambdas = self.add_weight(shape=(self.num_lambdas, 1),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True,
                                       name='lambdas',
                                       dtype='float32')
        if self.use_bias:
            self.b = self.add_weight(shape=(1, input_shape[-1]),
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     trainable=True,
                                     name='bias',
                                     dtype='float32')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = tf.constant(inputs, tf.float32)
        # Make sure input shape corresponds to convention of (batch size, timestamps, features)
        assert x.get_shape().ndims == 3, f'Inputs shape must be of form (batch size, #timestamps, #features, ' \
                                         f'yours is of form {x.get_shape().ndims}'

        x = denoise_tv_chambolle_nd(image=x, weights=self.lambdas, max_num_iter=self.max_iters)

        # Apply bias accordingly to resulted shape
        if self.use_bias:
            x = x + self.b

        # And activation
        if self.activation is not None:
            x = self.activation(x)

        return x
