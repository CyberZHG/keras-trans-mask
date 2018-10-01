import keras


class RemoveMask(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        return inputs


class RestoreMask(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(RestoreMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def call(self, inputs, **kwargs):
        return inputs[0]
