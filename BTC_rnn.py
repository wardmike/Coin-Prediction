import tensorflow as tf
import numpy as np
import random
import time
import sys


random.seed(time.time())



class currency_data(object):
	def __init__(self, currency_name, input_size):
		self.currency_name = currency_name
		self.input_size = input_size
		self.normalized = True
		self.num_steps = 30
		self.test_ratio = 0.1

		self.price_data = self.readInputFile()

		self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.price_data)







	def readInputFile(self):
		inputFile = open("data/5-minute/" + self.currency_name + ".txt")

		input_data = []
		for line in inputFile:
			vals = line.split("|")
			input_data.append(float(vals[3]))

		return np.array(input_data)


	def _prepare_data(self, seq):
		# split into items of input_size
#		newSeq = []
#		for i in range(len(seq) // self.input_size):
#			newSeq.append(np.array(seq[i * self.input_size: (i + 1) * self.input_size]))

#		seq = newSeq

		seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size]) for i in range(len(seq) // self.input_size)]

		if self.normalized:
			seq = [seq[0] / seq[0][0] - 1.0] + [
				curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

		# split into groups of num_steps
		X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
		y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

		train_size = int(len(X) * (1.0 - self.test_ratio))
		train_X, test_X = X[:train_size], X[train_size:]
		train_y, test_y = y[:train_size], y[train_size:]
		return train_X, train_y, test_X, test_y


	def generate_one_epoch(self, batch_size):
		num_batches = int(len(self.train_X)) // batch_size
		if batch_size * num_batches < len(self.train_X):
			num_batches += 1

		batch_indices = range(num_batches)
		random.shuffle(batch_indices)
		for j in batch_indices:
			batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
			batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
			assert set(map(len, batch_X)) == {self.num_steps}
			yield batch_X, batch_y




#easily configure values up-top
class RNNConfig(object):
    input_size=1
    num_steps=30
    lstm_size=128
    num_layers=1
    keep_prob=0.8
    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 1
    max_epoch = 50

    def __init__(self, currency_name):
    	self.currency_name = currency_name

config = RNNConfig(sys.argv[1])


#values to use in session
inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size])
targets = tf.placeholder(tf.float32, [None, config.input_size])
learning_rate = tf.placeholder(tf.float32, None)

#creates a cell, line below creates multi-cells using this function if num_layers > 1
def _create_one_cell():
	return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
	if config.keep_prob < 1.0:
		return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)


cell = tf.contrib.rnn.MultiRNNCell(
        [_create_one_cell() for _ in range(config.num_layers)], 
        state_is_tuple=True
    ) if config.num_layers > 1 else _create_one_cell()



val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

# Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
# After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
val = tf.transpose(val, [1, 0, 2])
# last.get_shape() = (batch_size, lstm_size)
last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

#generates weight and bias, then predicts value
weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.input_size]))
bias = tf.Variable(tf.constant(0.1, shape=[config.input_size]))
prediction = tf.matmul(last, weight) + bias

#checks for error, optimizes for error
loss = tf.reduce_mean(tf.square(prediction - targets))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

#get a dataset for testing, then test
currency_data_set = currency_data(config.currency_name, config.input_size)


test_data_feed = {
	learning_rate: 0.0,
	inputs: currency_data_set.test_X,
	targets: currency_data_set.test_y
}




with tf.Session() as sess:
	tf.global_variables_initializer().run()

	global_step = 0

	sumTestLoss = 0
	numberOfTestLosses = 0

	for epoch_step in range(config.max_epoch):
		current_lr = config.init_learning_rate
		if epoch_step == config.max_epoch-1:
			current_lr = 0.0

		for batch_X, batch_y in currency_data(config.currency_name, config.input_size).generate_one_epoch(config.batch_size):
			global_step += 1
			train_data_feed = {
				inputs: batch_X, 
				targets: batch_y, 
				learning_rate: current_lr
			}

			train_loss, _ = sess.run([loss, minimize], train_data_feed)
			if epoch_step == config.max_epoch-1:
				test_loss, test_pred = sess.run([loss, prediction], test_data_feed)
				numberOfTestLosses += 1
				sumTestLoss += test_loss


		print "Epoch " + str(epoch_step) + " completed."

	averageTestLoss = sumTestLoss/numberOfTestLosses
	print "Average Test Loss was: ", averageTestLoss









