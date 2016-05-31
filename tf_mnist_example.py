import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import multivariate_normal as MVN

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1], padding='SAME')

with tf.device('/gpu:0'):
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 1])

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 1])
	b_fc2 = bias_variable([1])

	logit = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_conv= 1.0 / (1.0 + tf.exp(-logit))

	y_conv = tf.clip_by_value(y_conv, 0.00001, 0.99999)

	# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	cross_entropy = (-1.0/256.0) * tf.reduce_sum(
		y_ * tf.log(y_conv) + (1.0 - y_) * tf.log(1.0 - y_conv)
	)

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ------------------------------------
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	batch_size = 128

	labels = np.concatenate(
		(
			np.ones(batch_size),
			np.zeros(batch_size)
		),
		axis=0
	).astype(np.float32)
	labels = np.expand_dims(labels, 1)

	print('LABELS SHAPE')
	print(labels.shape)

	sess.run(tf.initialize_all_variables())

	# Train Loop ------------------------------------------------------------
	for i in range(20000):
		batch = mnist.train.next_batch(batch_size)
		digits = batch[0]
		noise = MVN(
			np.zeros(784),
			np.diag(np.ones(784)),
			size = batch_size
		)
		batch_input = np.concatenate(
			(digits, noise),
			axis=0
		)

		if i%1 == 0:
			result = sess.run([cross_entropy, y_conv], feed_dict={
				x: batch_input, y_: labels, keep_prob: 1.0})
			print("step %d, training CE %g, training Acc. %g"%(i, result[0], np.mean(np.round(result[1]) == labels)))
			# print(result[1])
		sess.run(train_step, feed_dict={x: batch_input, y_: labels, keep_prob: 1.0})
		# print(batch_input.shape)

	# print("test accuracy %g"%accuracy.eval(feed_dict={
	# 	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))