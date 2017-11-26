
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataset/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + bias)
actual = tf.placeholder(tf.float32, [None, 10])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=actual, logits=prediction)
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(12000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x:batch_x, actual: batch_y})

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print "Accuracy of prediction of digits is: %s" %(sess.run(accuracy*100, feed_dict={x: mnist.test.images, actual: mnist.test.labels}))
