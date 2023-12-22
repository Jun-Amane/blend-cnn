import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiheadAttention(keras.Model):
    def __init__(self, input_size, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        self.W_q = layers.Dense(units=input_size)
        self.W_k = layers.Dense(units=input_size)
        self.W_v = layers.Dense(units=input_size)

        self.plus = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        q = tf.reshape(self.W_q(inputs), (tf.shape(inputs)[0], -1, self.num_heads, self.head_size))
        k = tf.reshape(self.W_k(inputs), (tf.shape(inputs)[0], -1, self.num_heads, self.head_size))
        v = tf.reshape(self.W_v(inputs), (tf.shape(inputs)[0], -1, self.num_heads, self.head_size))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attn_weights = tf.linalg.matmul(q, k, transpose_b=True) / self.head_size ** 0.5
        attn_weights = layers.Softmax()(attn_weights)

        attn_output = tf.linalg.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (tf.shape(inputs)[0], -1, self.num_heads * self.head_size))
        attn_output = tf.squeeze(attn_output, axis=1)
        tf.ensure_shape(attn_output, (None, 1024))

        res_output = self.plus([inputs, attn_output])
        # res_output = inputs + attn_output
        norm_output = self.layer_norm(res_output)

        return norm_output
