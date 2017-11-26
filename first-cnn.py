import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("dataset/", one_hot=True)
image_pixels = 28
kernel = 5


def weight_initialise(shape):
    value = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(value)


def bias_initialise(shape):
    value = tf.constant(0.1, shape=shape)
    return tf.Variable(value)

def classifier(x):
   
    input_layer = tf.reshape(x, [-1, image_pixels, image_pixels, 1])

    params_conv1 = weight_initialise([kernel, kernel, 1, 32])

    bias_conv1 = bias_initialise([32])

    output_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, params_conv1, strides=[1, 1, 1, 1], padding='SAME')+bias_conv1)

    pool_conv1 = tf.nn.max_pool(output_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
    params_conv2 = weight_initialise([kernel, kernel, 32, 64])
    bias_conv2 = bias_initialise([64])

    output_conv2 = tf.nn.relu(tf.nn.conv2d(pool_conv1, params_conv2, strides=[1, 1, 1, 1], padding='SAME')+bias_conv2)

    pool_conv2 = tf.nn.max_pool(output_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    params_fc1 = weight_initialise([7*7*64, 1024])
    bias_fc1 = bias_initialise([1024])
    pool_conv2_flat = tf.reshape(pool_conv2, [-1, 7*7*64])
    output_fc1 = tf.nn.relu(tf.matmul(pool_conv2_flat, params_fc1) + bias_fc1)

    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(output_fc1, keep_prob)

    params_fc2 = weight_initialise([1024, 10])
    bias_fc2 = bias_initialise([10])
    y_conv = tf.matmul(dropout, params_fc2) + bias_fc2

    return y_conv, keep_prob


# Declaring variables

x = tf.placeholder(tf.float32, [None, 784])
targets = tf.placeholder(tf.float32, [None, 10])
y_conv, keep_prob = classifier(x)

# Calculating loss

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(targets, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(50)
    sess.run(train, feed_dict={x: batch_x, targets: batch_y, keep_prob: 0.5})

print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, targets: mnist.test.labels, keep_prob: 1.0}))

