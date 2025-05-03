import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from src.models.base import BaseModel
from src.metrics import compute_metrics

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.do1 = layers.Dropout(rate)
        self.do2 = layers.Dropout(rate)

    def call(self, x, training=False):
        att = self.att(x, x)
        att = self.do1(att, training)
        x = self.ln1(x + att)
        ff = self.ffn(x)
        ff = self.do2(ff, training)
        return self.ln2(x + ff)

class CustomClassifierKeras(keras.Model, BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vectorizer = TextVectorization(max_tokens=config.get('vocab_size',20000),
                                            output_sequence_length=config['max_length'])
        self.embedding = layers.Embedding(
            input_dim=config.get('vocab_size',20000),
            output_dim=config.get('embed_dim',128)
        )
        self.transformer = TransformerBlock(
            config.get('embed_dim',128),
            config.get('num_heads',4),
            config.get('ff_dim',256),
            rate=config.get('dropout_rate',0.1)
        )
        self.pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(config.get('dropout_rate',0.1))
        self.dense = layers.Dense(config.get('dense_units',64), activation='relu')
        self.clf = layers.Dense(config['num_labels'], activation='softmax')

    def train_model(self, train_texts, train_labels, val_texts, val_labels, config, log_callback=False):
        train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
        train_ds = train_ds.map(lambda x,y: (self.vectorizer(x), y)).batch(config['batch_size']).prefetch(1)
        val_ds   = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
        val_ds   = val_ds.map(lambda x,y: (self.vectorizer(x), y)).batch(config['batch_size']).prefetch(1)
        self.compile(
            optimizer=keras.optimizers.Adam(config['lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history = self.fit(train_ds, validation_data=val_ds, epochs=config['epochs'], verbose=0)
        if log_callback:
            for e, loss in enumerate(history.history['loss']):
                print(f"Epoch {e+1}, Loss: {loss:.4f}")
        self._last_loss = history.history['loss'][-1]

    def predict(self, texts):
        ds = tf.data.Dataset.from_tensor_slices(texts).map(self.vectorizer).batch(self.config['batch_size'])
        logits = self(ds)
        return tf.argmax(logits, axis=1).numpy().tolist()

    def save_state(self, path):
        self.save_weights(path + '.h5')

    def evaluate_model(self, texts, labels):
        preds = self.predict(texts)
        return compute_metrics(labels, preds)
