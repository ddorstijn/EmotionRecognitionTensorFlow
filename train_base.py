import json
import collections
import os
import math

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    def __init__(self, data, labels, dtype=dtypes.float32,
                 reshape=True, seed=None):
        """Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`, or
        `float32` to rescale into `[0, 1]`. Seed arg provides for convenient
        deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is
        # returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32'
                            % dtype)
        assert data.shape[0] == labels.shape[0], (
            'data.shape: %s labels.shape: %s' %
            (data.shape, labels.shape))
        self._num_examples = data.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # if reshape:
        #     assert data.shape[3] == 1
        #     data = data.reshape(data.shape[0],
        #                         data.shape[1] * data.shape[2])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            data = data.astype(np.float32)
            data = np.multiply(data, 1.0 / 255.0)
            self._data = data
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((data_rest_part, data_new_part),
                                  axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]


def inference(data, input_size, hidden1_units, hidden2_units, output_size):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      data: data placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([input_size, hidden1_units],
                                stddev=1.0 / math.sqrt(float(input_size))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, output_size],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')

        biases = tf.Variable(tf.zeros([output_size]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, output_size].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def to_numpy(data, labels):
    """ Small helper function to convert list of arrays to a 2D array """
    return np.vstack(data), np.array(labels)


def read_data_sets(input_dir, dtype=dtypes.float32, reshape=True, seed=None):
    """ Load data from the JSON files which contain skeleton information """
    temp_train = []
    temp_labels = []
    test_data = []
    test_labels = []

    for folder, subs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(folder, file)
                with open(json_file) as json_data:
                    if folder.split('\\')[-2] == "train":
                        data_train, data_label = prep_json(
                            json.load(json_data))
                        temp_train.append(data_train)
                        temp_labels.append(data_label)
                    else:
                        data_train, data_labels = prep_json(
                            json.load(json_data))
                        test_data.append(data_train)
                        test_labels.append(data_labels)

    split = np.int(len(temp_train) * 0.7)
    train_data, val_data = temp_train[:split], temp_train[split:]
    train_labels, val_labels = temp_labels[:split], temp_labels[split:]

    # Convert to numpy arrays
    train_data, train_labels = to_numpy(train_data, train_labels)
    val_data, val_labels = to_numpy(val_data, val_labels)
    test_data, test_labels = to_numpy(test_data, test_labels)

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_data, train_labels, **options)
    validation = DataSet(val_data, val_labels, **options)
    test = DataSet(test_data, test_labels, **options)

    return Datasets(train=train, validation=validation, test=test)


def prep_json(data):
    """ Prepare JSON data from files """
    label = data.get("label")

    if label == "angry":
        label = 0
    elif label == "boredom":
        label = 1
    elif label == "disgust":
        label = 2
    elif label == "fear":
        label = 3
    elif label == "neutral":
        label = 4
    elif label == "sadness":
        label = 5
    elif label == "surprise":
        label = 6

    return np.asarray(data.get("pose")), label
