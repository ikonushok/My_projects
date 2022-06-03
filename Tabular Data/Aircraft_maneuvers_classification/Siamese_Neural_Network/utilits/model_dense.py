import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, MaxPooling2D, Concatenate, Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.regularizers import l2

# сделаем так, чтобы tf не резервировал под себя сразу всю память
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# фиксируем сиды
# https://coderoad.ru/51249811/Воспроизводимые-результаты-в-Tensorflow-с-tf-set_random_seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)  # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
tf.random.set_seed(seed)


"""
### Create the Embedding Model
This model takes in input image and generates some `emb_size`- dimensional embedding for the image, 
plotted on some latent space.
The untrained model's embedding space is stored for later use when comparing clustering 
between the untrained and the trained model using PCA, based on 
[this notebook](https://github.com/AdrianUng/keras-triplet-loss-mnist/blob/master/Triplet_loss_KERAS_semi_hard_from_TF.ipynb).
"""
def create_embedding_model(emb_size, x_train_w_h):
    embedding_model = tf.keras.models.Sequential([
        Dense(28*28,#4096,
              activation='relu',
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform',
              input_shape=(x_train_w_h,)),
        Dense(emb_size,
              activation=None,
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform')
    ])

    embedding_model.summary()

    return embedding_model


"""
### Create the SNN
This model takes a triplet image input, 
passes them to the embedding model for embedding, then concats them together for the loss function
"""
def create_SNN(network, x_train_w_h):
    input_anchor = tf.keras.layers.Input(shape=(x_train_w_h,))
    input_positive = tf.keras.layers.Input(shape=(x_train_w_h,))
    input_negative = tf.keras.layers.Input(shape=(x_train_w_h,))

    embedding_anchor = network(input_anchor)
    embedding_positive = network(input_positive)
    embedding_negative = network(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive,
                                          embedding_negative], axis=1)

    siamese_net = tf.keras.models.Model([input_anchor, input_positive, input_negative],
                                        output)
    siamese_net.summary()

    return siamese_net