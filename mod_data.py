import numpy as np 
from random import shuffle

filename = "/home/stan/Projects/TensorFlow/coin-prediction/data/condensed/data_just_24h.csv"
offset = 288

def sample_handling(sample):
	featureset = []
	input_data = []
	output_data = []
	global offset
	with open(sample, 'r') as f:
		contents = f.readlines()
		hm_lines = len(contents)-offset
		for l in contents[1:hm_lines]:
			l = l.strip().split(',')
			input_data.append(l)
		for l in contents[offset+1:len(contents)]:
			l = l.strip().split(',')
			output_data.append(l)
		for i in range(len(input_data)):
			featureset.append([input_data[i], output_data[i]])
		print(len(featureset))
	return featureset

def create_feature_sets_and_labels(test_size = .1):
	global filename
	featureset = []
	featureset += sample_handling(filename)

	shuffle(featureset)

	testing_size = int(test_size * len(featureset))
	
	print(testing_size)

	train_x = [item[0] for item in featureset[:-testing_size]]
	train_y = [item[1] for item in featureset[:-testing_size]]

	test_x = [item[0] for item in featureset[-testing_size:]]
	test_y = [item[1] for item in featureset[-testing_size:]]

	return train_x,train_y,test_x,test_y

#create_feature_sets_and_labels()