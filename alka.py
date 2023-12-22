import alkaset
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ImageResnet import ResNet18
from TextCNN import TextCNN
from multihead_attention import MultiheadAttention


def normalize_img(input, label):
    image = input['image']
    image = tf.cast(image, tf.float32) / 255.
    input['image'] = image
    return input, label


VOCAB_SIZE = 7000
MAX_SEQ_LEN = 250
#
# binary_vectorize_layer = layers.TextVectorization(
#     max_tokens=VOCAB_SIZE,
#     output_mode='binary'
# )
int_vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQ_LEN,
    pad_to_max_tokens=True
)


def get_text(input, label):
    return input['descriptions']


train = tfds.load('alkaset', split='train', as_supervised=True)
train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

train_text = train.map(get_text)
int_vectorize_layer.adapt(train_text)

train = train.batch(32)
train = train.cache()
train = train.prefetch(tf.data.AUTOTUNE)


class ALKA(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_model = ResNet18()
        self.text_model = TextCNN(vocab_size=7000, vectorizer=int_vectorize_layer)

        self.attention = MultiheadAttention(input_size=1024, num_heads=8)

        self.feed_forward_network = keras.Sequential([
            keras.layers.Dropout(0.3),
            keras.layers.Dense(102)
        ])

    def call(self, inputs, training=None, mask=None):
        image = inputs['image']
        text = inputs['descriptions']

        image_features = self.image_model(image)
        tf.ensure_shape(image_features, (None, 512))
        text_features = self.text_model(text)
        tf.ensure_shape(text_features, (None, 512))

        concat_features = tf.concat([image_features, text_features], axis=1)
        tf.ensure_shape(concat_features, (None, 1024))
        attn_out = self.attention(concat_features)
        ffn_out = self.feed_forward_network(attn_out)
        return ffn_out


#
net = ALKA()

net.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# net.run_eagerly = True

net.fit(train, epochs=10)
net.summary()
net.evaluate(train)
