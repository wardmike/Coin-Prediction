

import tensorflow as tf
from tensorflow.contrib import rnn
import os
#will be reading in 28px x 28px to get the machine to understand what is going on

# input > weight > hidden layer 1(activation function) > weights > hidden layer 2
#hidden layer 2 will develop an activation function > weights > ouput layer

#compare machine output to intended output > cost function

#opmtimizer function > minimize cost(AdamOptimizer..... SGD, AdaGrad) has multiple optimizing functions

#backpropogation: where we work backwards and adjust the weights

#feed forward + backpropogation = epoch

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

filename = "data_just_1h_scaled.csv"

features = tf.placeholder(tf.int32, shape=[3], name='features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')
# 10 classes, 0-9
# where if input 0, output 0

hm_epochs = 3
n_classes = 10
batch_size = 128 # will do batches of 100 images at a time
#do this to not take up all of your memory

chunk_size = 28
n_chunks = 28
rnn_size = 512

x = tf.placeholder('float', [None, n_chunks, chunk_size]) #second parameter defines input for x, if doesn't fit paramater, error will be thrown
y = tf.placeholder('float')

def recurrent_neural_network_model(x):

	# (input_data * weights) + biases --> biases adjust 0's from imput

	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)


	output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

	return output

def train_neural_network(x):

	prediction = recurrent_neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	#	alternate for minimize			learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + backpropogation

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# nested for loops are simply training our algorithms

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)



		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}))

train_neural_network(x)

