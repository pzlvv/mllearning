import tensorflow as tf
import numpy as np

train_X = np.matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
n_input = 4
n_hidden_1 = 2

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    'decoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]))
}

biases = {
    'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1, 1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_input, 1]))
}

x = tf.placeholder(tf.float32, [n_input, 4])


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(
        # tf.matmul(tf.transpose(x), weights['decoder_h1']),
        tf.matmul(weights['decoder_h1'], x),
        biases['decoder_h1']))
    return layer_1


def encoder(x):
    # Encoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(
        tf.matmul(weights['encoder_h1'], x),
        biases['encoder_h1']))
    return layer_1


encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = x

cost = tf.reduce_mean(tf.pow((y_true - y_pred), 2))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(weights["encoder_h1"]))
    for epoch in range(10000):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_X.T})
    print(sess.run(weights["encoder_h1"]))
    print(sess.run(y_pred, feed_dict={x: train_X.T}))
