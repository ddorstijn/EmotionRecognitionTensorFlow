import tensorflow as tf
import numpy as np
import os
import json

# Load
data_dir = os.path.abspath('Data')

# Check for existence
if not os.path.exists(data_dir):
    Exception('No data found')

# Number of neurons in each layer
input_num_units = 36
hidden_num_units = 32
output_num_units = 7

hidden_layer_count = 3

# Seed for random selection
seed = 128
random = np.random.RandomState(seed)

# Define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# Set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01


weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units],
                                           seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units],
                                           seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}


def load_data():
    for folder, subs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(folder, file)
                with open(json_file) as json_data:
                    data = json.load(json_data)
                    print(data.get("pose"))

load_data()


def create_batch(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = random.choice(dataset_length, batch_size)

    batch_x = eval(
        dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)

    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values

    return batch_x, batch_y

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# Create session and run the graph
with tf.Session() as sess:
    pass
    # Create initialized variables
#     sess.run(init)

#     # For each epoch, do:
#     # For each batch, do:
#     # Create pre-processed batch
#     # Run optimizer by feeding batch
#     # Find cost and reiterate to minimize

#     for epoch in range(epochs):
#         avg_cost = 0
#         total_batch = int(train.shape[0] / batch_size)
#         for i in range(total_batch):
#             batch_x, batch_y = batch_creator(
#                 batch_size, train_x.shape[0], 'train')
#             _, c = sess.run([optimizer, cost], feed_dict={
#                             x: batch_x, y: batch_y})

#             avg_cost += c / total_batch

#         print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

#     print("\nTraining complete!")

#     # Find predictions on val set
#     pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
#     print("Validation Accuracy:",
#           accuracy.eval({
#               x: val_x.reshape(-1, input_num_units),
#               y: val_y}))

#     predict = tf.argmax(output_layer, 1)
#     pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

# # Close session
# sess.close()
