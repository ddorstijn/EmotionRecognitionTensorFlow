
import tensorflow as tf
input_size = 36


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
        tf.float32, shape=(batch_size, input_size))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return data_placeholder, labels_placeholder
