from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init_(self):

		random.seed(1)

		self.synaptic_weights = 2 * random.random((34, 1)) - 

	def __sigmoid(self, x):
		return 1/(1+exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1-x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			output = self.think(training_set_inputs)

			error = training_set_outputs - output

			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += adjustment


	def think(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

	def readInputFile(self, fileName):
		inputFile = open(fileName)

		next(inputFile)
		input_prices = []
		for line in inputFile:
			prices = map(float, line.split(","))
			input_prices.append(prices)

		return array(input_prices)

	def readOutputFile(self, fileName):
		outputFile = open(fileName)

		output_value = []
		for line in outputFile:
			output_value.append(int(line))

		return array(output_value)







if __name__ == "__main__":

	neural_network = NeuralNetwork()

	print "Random Starting Weights: "
	print neural_network.synaptic_weights

	training_set_inputs = neural_network.readInputFile("data_just_prices.csv")
	training_set_outputs = neural_network.readOutputFile("trend_data_bitcoin.txt").T

	neural_network.train(training_set_inputs, training_set_outputs, 50)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights



	