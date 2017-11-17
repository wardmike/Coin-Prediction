import tensorflow as tf

#will be reading in 28px x 28px to get the machine to understand what is going on

# input > weight > hidden layer 1(activation function) > weights > hidden layer 2
#hidden layer 2 will develop an activation function > weights > ouput layer

#compare machine output to intended output > cost function

#opmtimizer function > minimize cost(AdamOptimizer..... SGD, AdaGrad) has multiple optimizing functions

#backpropogation: where we work backwards and adjust the weights

#feed forward + backpropogation = epoch

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# 10 classes, 0-9
# where if input 0, output 0

n_nodes_hl1 = 500 #each of these are hidden layers
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # will do batches of 100 images at a time
#do this to not take up all of your memory

x = tf.placeholder('float', [None, 784]) #second parameter defines input for x, if doesn't fit paramater, error will be thrown
y = tf.placeholder('float')

def neural_network_model(data):

	# (input_data * weights) + biases --> biases adjust 0's from imput

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output

def train_neural_network(x):

	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	#	alternate for minimize			learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + backpropogation
	hm_epochs = 2

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# nested for loops are simply training our algorithms

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)



		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

