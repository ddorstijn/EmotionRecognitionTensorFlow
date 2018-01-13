# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import json
import collections

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class DataSet(object):
  def __init__(self, skeleton, labels, dtype=dtypes.float32, reshape=True, seed=None):
    """Construct a DataSet.
    `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32`
    to rescale into `[0, 1]`. Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
        raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                        dtype)

    assert skeleton.shape[0] == labels.shape[0], (
        'skeleton.shape: %s labels.shape: %s' % (skeleton.shape, labels.shape))
    self._num_examples = skeleton.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # if reshape:
    #     assert skeleton.shape[3] == 1
    #     skeleton = skeleton.reshape(skeleton.shape[0],
    #                                 skeleton.shape[1] * skeleton.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        print(skeleton)
        skeleton = skeleton.astype(np.float32)
        skeleton = np.multiply(skeleton, 1.0 / 255.0)
        self._skeleton = skeleton
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

  @property
  def skeleton(self):
    return self._skeleton

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
      self._skeleton = self.skeleton[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      skeleton_rest_part = self._skeleton[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._skeleton = self.skeleton[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      skeleton_new_part = self._skeleton[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((skeleton_rest_part, skeleton_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._skeleton[start:end], self._labels[start:end]


def load_json(dtype=dtypes.float32, reshape=True, seed=None):
    """ Load data from the JSON files which contain skeleton information """
    temp_train = []
    temp_labels = []
    test_train = []
    test_labels = []

    for folder, subs, files in os.walk(FLAGS.input_data_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(folder, file)
                with open(json_file) as json_data:
                    if folder.split('\\')[-2] == "train":
                        data_train, data_label = prep_json(json.load(json_data))
                        temp_train.append(data_train)
                        temp_labels.append(data_label)
                    else:
                        data_train, data_labels = prep_json(json.load(json_data))
                        test_train.append(data_train)
                        test_labels.append(data_labels)

    split_point = np.int(len(temp_train)*0.7)
    train_data, val_data = temp_train[:split_point], temp_train[split_point:]
    train_labels, val_labels = temp_labels[:split_point], temp_labels[split_point:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(np.asarray(train_data), np.asarray(train_labels), **options)
    validation = DataSet(np.asarray(validation_data), np.asarray(validation_labels), **options)
    test = DataSet(np.asarray(test_data), np.asarray(test_labels), **options)

    return Datasets(train=train, validation=validation, test=test)

def prep_json(data):
    """ Prepare JSON data from files """
    label = data.get("label")

    if label == "angry":
        label = [1,0,0,0,0,0,0]
    elif label == "boredom":
        label = [0,1,0,0,0,0,0]
    elif label == "disgust":
        label = [0,0,1,0,0,0,0]
    elif label == "fear":
        label = [0,0,0,1,0,0,0]
    elif label == "neutral":
        label = [0,0,0,0,1,0,0]
    elif label == "sadness":
        label = [0,0,0,0,0,1,0]
    elif label == "surprise":
        label = [0,0,0,0,0,0,1]

    return data.get("pose"), label


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.

    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
  print(data_sets)
  data_sets = load_json()
  print(data_sets)
  exit()

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.abspath('data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
