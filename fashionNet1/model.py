#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    15-Aug-2025 14:30:44

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    imageinput_unnormalized = keras.Input(shape=(28,28,1), name="imageinput_unnormalized")
    imageinput = keras.layers.Normalization(axis=(1,2,3), name="imageinput_")(imageinput_unnormalized)
    imageinputperm = layers.Permute((3,2,1))(imageinput)
    flatten = layers.Flatten()(imageinputperm)
    fc = layers.Dense(128, name="fc_")(flatten)
    relu = layers.ReLU()(fc)
    fc_1 = layers.Dense(10, name="fc_1_")(relu)
    softmax = layers.Softmax()(fc_1)

    model = keras.Model(inputs=[imageinput_unnormalized], outputs=[softmax])
    return model
