# pylint: disable=missing-docstring
import os
import sys
import tensorflow as tf
import shared
import train_base as base
import math

data_placeholder, labels_placeholder = shared.placeholder_inputs(12)
new_saver = tf.train.import_meta_graph(os.path.join("save", 'model.ckpt.meta'))
graph = tf.get_default_graph();

def inference(data, input_size, hidden1_units, hidden2_units, output_size):
    """Build the data model up to where it may be used for inference.

    Args:
      data: data placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    weights = graph.get_tensor_by_name('hidden1/weights:0');
    biases = graph.get_tensor_by_name('hidden1/biases:0');
    hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
    
    weights = graph.get_tensor_by_name('hidden2/weights:0');
    biases = graph.get_tensor_by_name('hidden2/biases:0');
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    weights = graph.get_tensor_by_name('softmax_linear/weights:0');
    biases = graph.get_tensor_by_name('softmax_linear/biases:0');
    logits = tf.matmul(hidden2, weights) + biases
    return logits

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

with tf.Session() as sess:
    new_saver.restore(sess,tf.train.latest_checkpoint("save"))
    
    # hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)

    global_step_tensor = graph.get_tensor_by_name('global_step:0')
    print(sess.run(global_step_tensor))
    print(tf.global_variables())
    
    i = 0;
    # for n in tf.get_default_graph().as_graph_def().node:
    #     print(n.name)
    #     print(n)
    #     i+=1
    #     if i > 4:
    #         sys.exit()

    # print(graph.get_tensor_by_name("Placeholder:0"))
    # print(graph.get_tensor_by_name("Placeholder_1:0"))
    # print(graph.get_tensor_by_name("Placeholder_2:0"))
    # print(graph.get_tensor_by_name("Placeholder_1_1:0"))
    # sess.run(graph.get_tensor_by_name('Placeholder:0'));