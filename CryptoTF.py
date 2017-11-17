

import tensorflow as tf

crypto = 

#number variables nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#number of epochs
num_epochs = 5

#read-in size and ouput size
in_size = 171
out_size = 34

#batch_size
batch_size = 20

x = tf.placeholder('float', [None, in_size])
y = tf.placeholder('float')

def tf_model(data):

	hl1 = {'weights':tf.Variables(tf.random_normal([in_size, n_nodes_hl1])),
		   'biases':tf.Variables(tf.random_normal([n_nodes_hl1]))}
	
	hl2 = {'weights':tf.Variables(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
		   'biases':tf.Variables(tf.random_normal([n_nodes_hl2]))}
	
	hl3 = {'weights':tf.Variables(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
		   'biases':tf.Variables(tf.random_normal([n_nodes_hl3]))}

	output = {'weights':tf.Variables(tf.random_normal([n_nodes_hl3, out_size])),
		   'biases':tf.Variables(tf.random_normal([out_size]))}

	l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
	l2 = tf.nn.relu(l1)

	l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
	l3 = tf.nn.relu(l1)

	output = tf.add(tf.matmul(l3, output['weights']), output['biases'])

	return output

def train_net(x):
	prediction = tf_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs)
			tot_cost = 0
			for _ in range(int(crypto.train.num_examples/batch_size)):
				epoch_x, epoch_y = crypto.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				tot_cost += c
			print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', tot_cost)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accura)
