# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import train_base as base

import tensorflow as tf
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY as DEFAULT_SIG_DEF

input_size = 36
output_size = 7

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.

    Returns:
    data_placeholder: Data placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    data_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, input_size), name="input")
    labels_placeholder = tf.placeholder(
        tf.int32, shape=(batch_size), name="output_size")
    return data_placeholder, labels_placeholder


def fill_feed_dict(data_set, data_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    data_pl: The data placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    data_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        data_pl: data_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess, eval_correct, data_placeholder,
            labels_placeholder, data_set):
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
                                   data_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))


def run_training():
    """Train Data for a number of steps"""
    # Get the sets of images and labels for training, validation, and tests
    data_sets = base.read_data_sets(FLAGS.input_data_dir)

    # Generate placeholders for the images and labels.
    data_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model
    logits = base.inference(data_placeholder,
                            input_size,
                            FLAGS.hidden1,
                            FLAGS.hidden2,
                            output_size)

    # Add to the Graph the Ops for loss calculation
    loss = base.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients
    train_op = base.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation
    eval_correct = base.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries
    summary = tf.summary.merge_all()

    # Add the variable initializer Op
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.export_dir)

    # Tell TensorFlow that the model will be built into the default Graph
    with tf.Session() as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Run the Op to initialize the variables
        sess.run(init)

        # Start the training loop.
        step = 0
        while step < 2000:
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step
            feed_dict = fill_feed_dict(data_sets.train,
                                       data_placeholder,
                                       labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often
            if step % 100 == 0:
                # Print status to stdout
                print(
                    'Step %d: loss = %.2f (%.3f sec)' %
                    (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or loss_value == 0.0:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        data_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        data_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        data_placeholder,
                        labels_placeholder,
                        data_sets.test)
            if loss_value == 0.0:
                break

            step += 1

        output = sess.graph.get_tensor_by_name('output')
        input_tensor = sess.graph.get_tensor_by_name('input')
        sig_def = tf.saved_model.signature_def_utils.predict_signature_def(
            {'input': input_tensor},
            {'output': output}
        )
        builder.add_meta_graph_and_variables(
            sess, tf.saved_model.tag_constants.SERVING,
            signature_def_map={
                DEFAULT_SIG_DEF: sig_def
            }
        )


def main(_):
    # Make sure the directory is empty so it can be used to save the data
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
        default=12,
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
        default=os.path.abspath('logs'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        default=os.path.abspath('export'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
