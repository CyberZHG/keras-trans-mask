from unittest import TestCase
import os
import tempfile

import numpy as np

from keras_trans_mask.backend import keras
from keras_trans_mask import CreateMask, RemoveMask, RestoreMask


class TestMasks(TestCase):

    def test_over_fit(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = keras.layers.Embedding(
            input_dim=10,
            output_dim=15,
        )(input_layer)
        mask_layer = CreateMask(mask_value=9)(input_layer)
        embed_layer = RestoreMask()([embed_layer, mask_layer])
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
            [1, 2, 3, 4, 5, 9, 9, 9],
            [6, 7, 8, 9, 9, 9, 9, 9],
        ] * 1024)
        y = np.array([[0], [1]] * 1024)
        model_path = os.path.join(tempfile.gettempdir(), 'test_trans_mask_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'CreateMask': CreateMask,
            'RemoveMask': RemoveMask,
            'RestoreMask': RestoreMask,
        })
        model.fit(x, y, epochs=10)
