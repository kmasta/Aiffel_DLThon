import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerClassifier:
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, num_classes=5):
        self.tokenizer = None
        self.START_TOKEN = None
        self.END_TOKEN = None
        self.VOCAB_SIZE = None
        self.MAX_LENGTH = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        self.model = None

    def build_tokenizer(self, sentences, vocab_size=2**13):
        print("🔄 토크나이저 생성 중...")
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            sentences, target_vocab_size=vocab_size)
        self.START_TOKEN = [self.tokenizer.vocab_size]
        self.END_TOKEN = [self.tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2

    def encode(self, sentences):
        encoded = []
        for sentence in sentences:
            tokenized = self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN
            encoded.append(tokenized)
        return encoded

    def compute_max_length(self, encoded_sentences, percentile=95):
        lengths = [len(s) for s in encoded_sentences]
        self.MAX_LENGTH = int(np.percentile(lengths, percentile))

    def filter_and_pad(self, encoded_sentences, labels):
        filtered_sents, filtered_labels = [], []
        for s, l in zip(encoded_sentences, labels):
            if len(s) <= self.MAX_LENGTH:
                filtered_sents.append(s)
                filtered_labels.append(l)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            filtered_sents, maxlen=self.MAX_LENGTH, padding='post')
        return padded, np.array(filtered_labels)

    def create_model(self):
        inputs = layers.Input(shape=(self.MAX_LENGTH,))
        embedding_layer = layers.Embedding(self.VOCAB_SIZE, self.embed_dim)(inputs)
        x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)(embedding_layer)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
