import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TextCNN(keras.Model):
    def __init__(self, vocab_size, embed_dim=300, vectorizer=None, filter_sizes=None, num_filters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if num_filters is None:
            num_filters = [128, 128, 128, 128]
        if filter_sizes is None:
            filter_sizes = [3, 4, 5, 6]

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if vectorizer is not None:
            self.vectorizer = vectorizer

        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=self.embed_dim, mask_zero=True)
        self.conv_0 = layers.Conv1D(filters=num_filters[0], kernel_size=filter_sizes[0])
        self.conv_1 = layers.Conv1D(filters=num_filters[1], kernel_size=filter_sizes[1])
        self.conv_2 = layers.Conv1D(filters=num_filters[2], kernel_size=filter_sizes[2])
        self.conv_3 = layers.Conv1D(filters=num_filters[3], kernel_size=filter_sizes[3])

    def call(self, inputs, training=None, mask=None):

        # output shape: (batch_size, input_length, output_dim)
        x_vec = self.vectorizer(inputs)
        x_embed = self.embedding(x_vec)

        # x_reshaped = tf.transpose(x_embed, [0, 2, 1])

        x_conv_0 = layers.ReLU()(self.conv_0(x_embed))
        x_conv_1 = layers.ReLU()(self.conv_1(x_embed))
        x_conv_2 = layers.ReLU()(self.conv_2(x_embed))
        x_conv_3 = layers.ReLU()(self.conv_3(x_embed))

        x_pool_0 = layers.MaxPooling1D(pool_size=x_conv_0.shape[1])(x_conv_0)
        x_pool_1 = layers.MaxPooling1D(pool_size=x_conv_1.shape[1])(x_conv_1)
        x_pool_2 = layers.MaxPooling1D(pool_size=x_conv_2.shape[1])(x_conv_2)
        x_pool_3 = layers.MaxPooling1D(pool_size=x_conv_3.shape[1])(x_conv_3)

        x_pool_0 = tf.squeeze(x_pool_0, axis=1)
        x_pool_1 = tf.squeeze(x_pool_1, axis=1)
        x_pool_2 = tf.squeeze(x_pool_2, axis=1)
        x_pool_3 = tf.squeeze(x_pool_3, axis=1)

        x_fc = tf.concat([x_pool_0, x_pool_1, x_pool_2, x_pool_3], axis=1)

        return x_fc

