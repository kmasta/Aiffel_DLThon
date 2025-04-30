import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', num_classes=5, average='macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average

        self.true_positives = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.false_positives = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(tf.squeeze(y_true), tf.int64)

        for i in range(self.num_classes):
            y_i = tf.equal(y_true, i)
            p_i = tf.equal(y_pred, i)

            tp = tf.reduce_sum(tf.cast(y_i & p_i, tf.float32))
            fp = tf.reduce_sum(tf.cast(~y_i & p_i, tf.float32))
            fn = tf.reduce_sum(tf.cast(y_i & ~p_i, tf.float32))

            self.true_positives.assign(tf.tensor_scatter_nd_add(self.true_positives, [[i]], [tp]))
            self.false_positives.assign(tf.tensor_scatter_nd_add(self.false_positives, [[i]], [fp]))
            self.false_negatives.assign(tf.tensor_scatter_nd_add(self.false_negatives, [[i]], [fn]))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

        return tf.reduce_mean(f1) if self.average == 'macro' else f1

    def reset_state(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))
