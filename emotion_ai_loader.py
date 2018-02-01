# pylint: disable=missing-docstring
import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import json
import shared


def read_data_sets():
    """ Load data from the JSON files which contain skeleton information """
    train_data = []

    with open(os.path.abspath(FLAGS.input)) as json_file:
        data = json.load(json_file)
        train_data.append(np.asarray(data.get("pose")))

    return np.vstack(train_data)


def inference(data, input_size, hidden1_units, hidden2_units, output_size):
    """Build the data model up to where it may be used for inference.

    Args:
      data: data placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    graph = tf.get_default_graph()

    weights = graph.get_tensor_by_name('hidden1/weights:0')
    biases = graph.get_tensor_by_name('hidden1/biases:0')
    hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)

    weights = graph.get_tensor_by_name('hidden2/weights:0')
    biases = graph.get_tensor_by_name('hidden2/biases:0')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    weights = graph.get_tensor_by_name('softmax_linear/weights:0')
    biases = graph.get_tensor_by_name('softmax_linear/biases:0')
    logits = tf.matmul(hidden2, weights) + biases
    print(graph.get_tensor_by_name("softmax_linear/weights:0"))
    print("and: ")
    print(graph.get_operation_by_name("softmax_linear/weights"))

    return logits


def main(_):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(
            os.path.join("save", 'model.ckpt.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint("save"))

        data_placeholder = tf.placeholder(
            tf.float32, shape=(1, shared.input_size))
        logits = inference(data_placeholder, shared.input_size, 128, 32, 7)

        data_set = read_data_sets()
        data_set = data_set.astype(np.float32)
        data_set = np.multiply(data_set, 1.0 / 255.0)
        feed_dict = {data_placeholder: data_set}

        prediction = tf.argmax(logits, 1)
        best = sess.run([prediction], feed_dict)
        print(best)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default=os.path.abspath('test/Angry1.jpg.json'),
        help='Directory to put the input data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
