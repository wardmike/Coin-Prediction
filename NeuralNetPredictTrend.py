import numpy as np

class NeuralNetwork():
	def __init__(self):

		np.random.seed(1)

		self.synaptic_weights = 2 * np.random.random((34, 1)) - 1

	def __sigmoid(self, x):
		print 1/(1+np.exp(-x))
		return 1/(1+np.exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1-x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			output = self.think(training_set_inputs)

			error = training_set_outputs - output

			adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += adjustment


	def think(self, inputs):
		return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

	def readInputFile(self, fileName):
		inputFile = open(fileName)

		input_prices = []
		lines = inputFile.readlines()[1:]

		

		self.lastLine_inputFile = lines[-1]

		self.lastLine_prices = map(float, self.lastLine_inputFile.split(","))

		print self.lastLine_inputFile

		for line in lines[:-1]:
			prices = map(float, line.split(","))
			input_prices.append(prices)

		return np.array(input_prices, dtype = np.float256)

	def readOutputFile(self, fileName):
		outputFile = open(fileName)

		output_value_outer = []
		output_value = []
		lines = outputFile.readlines()[:-1]
		for line in lines:
			output_value.append(int(line))

		outputFile.close()

		output_value_outer.append(output_value)

		return np.array(output_value_outer)







if __name__ == "__main__":

	neural_network = NeuralNetwork()

	print "Random Starting Weights: "
	print neural_network.synaptic_weights

	training_set_inputs = neural_network.readInputFile("data_just_prices.csv")
	training_set_outputs = neural_network.readOutputFile("trend_data_bitcoin.txt").T

	neural_network.train(training_set_inputs, training_set_outputs, 10)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights

	print "Test on last line, should be a zero: "
	print neural_network.think(np.array(neural_network.lastLine_prices))



	