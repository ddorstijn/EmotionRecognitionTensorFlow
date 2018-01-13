import tensorflow as tf
import numpy as np
import os
import json

# Load
data_dir = os.path.abspath('data')
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
epochs = 100
batch_size = 8
learning_rate = 0.001


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
    """ """
    temp_data = {'data_x': [], 'data_y': []}
    test_data = {'test_x': [], 'test_y': []}

    for folder, subs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(folder, file)
                with open(json_file) as json_data:
                    if folder.split('\\')[-2] == "train":
                        data = prep_json(json.load(json_data))
                        temp_data['data_x'].append(data['data_x'])
                        temp_data['data_y'].append(data['data_y'])
                    else:
                        data = prep_json(json.load(json_data))
                        test_data['test_x'].append(data['data_x'])
                        test_data['test_y'].append(data['data_y'])


    split_point = np.int(len(temp_data['data_x'])*0.7)
    train_x, val_x = temp_data['data_x'][:split_point], temp_data['data_x'][split_point:]
    train_y, val_y = temp_data['data_y'][:split_point], temp_data['data_y'][split_point:]

    return {'train_x': train_x, 'train_y': train_y}, {'val_x': val_x, 'val_y': val_y}, test_data

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

    return {'data_x': data.get("pose"), 'data_y': label}

def create_batch(dataset_length, dataset):
    """Create batch with random samples and return appropriate format"""

    batch_mask = random.choice(dataset_length, batch_size)
    batch_x = []
    batch_y = []

    for i in batch_mask:
        batch_x.append(dataset['train_x'][i])
        batch_y.append(dataset['train_y'][i])

    batch_x = np.array(batch_x).reshape(batch_size, input_num_units)
    batch_y = np.array(batch_y).reshape(batch_size, output_num_units)

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
    #Create initialized variables
    sess.run(init)

    # For each epoch, do:
    # For each batch, do:
    # Create pre-processed batch
    # Run optimizer by feeding batch
    # Find cost and reiterate to minimize

    for epoch in range(epochs):
        avg_cost = 0
        train, validate, test = load_data()
        total_batch = int(len(train['train_x']) / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = create_batch(len(train), train)
            _, c = sess.run([optimizer, cost], feed_dict={
                            x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

    print("\nTraining complete!")

    # Find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print("Validation Accuracy:",
          accuracy.eval({
              x: np.asarray(validate['val_x'][0]).reshape(1, input_num_units),
              y: np.asarray(validate['val_y'][0]).reshape(1, output_num_units)
          }))

    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: np.asarray(test['test_x'])})
    print("Prediction is: ", pred[test_index])

# Close session
sess.close()
