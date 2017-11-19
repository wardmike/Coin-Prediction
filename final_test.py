import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from mod_data import create_feature_sets_and_labels

filename = "/home/stan/Projects/TensorFlow/coin-prediction/data/condensed/data_just_24h.csv"

train_x,train_y,test_x,test_y = create_feature_sets_and_labels()

hm_epochs = 5
n_classes = 34
batch_size = 132 

chunk_size = 17
n_chunks = 2
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

			i = 0
			while i < len(train_x):
				#print('within epoch', epoch,'i-value', i, 'still running')
				start = i
				end = i + batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size
				if i+batch_size > len(train_x):
					break
			print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:',epoch_loss)



		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:np.asarray(test_x).reshape((-1,n_chunks,chunk_size)), y:test_y}))

train_neural_network(x)

