from .backend import keras
from .backend import backend as K


class RemoveMask(keras.layers.Layer):
    """Remove mask from input tensor.

    # Input shape
        Tensor with shape: `(batch_size, ...)`.

    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        return K.identity(inputs)


class RestoreMask(keras.layers.Layer):
    """Restore mask from the second tensor.

    # Input shape
        Tensor with shape: `(batch_size, ...)`.
        Tensor with mask and shape: `(batch_size, ...)`.

    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, **kwargs):
        super(RestoreMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def call(self, inputs, **kwargs):
        return K.identity(inputs[0])
