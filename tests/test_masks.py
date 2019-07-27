from unittest import TestCase

import numpy as np

from keras_trans_mask.backend import keras
from keras_trans_mask import RemoveMask, RestoreMask


class TestMasks(TestCase):

    def test_over_fit(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = keras.layers.Embedding(
            input_dim=10,
            output_dim=15,
            mask_zero=True,
        )(input_layer)
        removed_layer = RemoveMask()(embed_layer)
        conv_layer = keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
        )(removed_layer)
        restored_layer = RestoreMask()([conv_layer, embed_layer])
        lstm_layer = keras.layers.LSTM(units=5)(restored_layer)
        dense_layer = keras.layers.Dense(units=2, activation='softmax')(lstm_layer)
        model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary()
        x = np.array([
            [1, 2, 3, 4, 5, 0, 0, 0],
            [6, 7, 8, 9, 0, 0, 0, 0],
        ] * 1024)
        y = np.array([[0], [1]] * 1024)
        model.fit(x, y, epochs=10)
