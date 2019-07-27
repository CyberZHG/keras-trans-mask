# Keras Transfer Masking

[![Travis](https://travis-ci.org/CyberZHG/keras-trans-mask.svg)](https://travis-ci.org/CyberZHG/keras-trans-mask)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-trans-mask/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-trans-mask)

Remove and restore masks for layers that do not support masking. Note that the result may be incorrect in most cases.

## Install

```bash
pip install keras-trans-mask
```

## Usage

`Conv1D` does not support masking. By removing the mask you'll get a "nearly correct" output:

```python
import keras
from keras_trans_mask import RemoveMask, RestoreMask

input_layer = keras.layers.Input(shape=(None,))
embed_layer = keras.layers.Embedding(
    input_dim=10,
    output_dim=15,
    mask_zero=True,
)(input_layer)
removed_layer = RemoveMask()(embed_layer)  # Remove mask from embeddings
conv_layer = keras.layers.Conv1D(
    filters=32,
    kernel_size=3,
    padding='same',
)(removed_layer)
restored_layer = RestoreMask()([conv_layer, embed_layer])  # Restore mask from embeddings
lstm_layer = keras.layers.LSTM(units=5)(restored_layer)
dense_layer = keras.layers.Dense(units=2, activation='softmax')(lstm_layer)
model = keras.models.Model(inputs=input_layer, outputs=dense_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()
```
