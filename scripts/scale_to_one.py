"""
Script that scales inputs to a scale from -1 to 1.

Finds the data point who's abolsute value is the largest.

That becomes 1 (or -1 if negative) and all the others are scaled around that.
"""



def main():
	fl = open("data_just_1h.csv")
	first_line = fl.readlines()[0]
	fl = open("data_just_1h.csv")
	lines = fl.readlines()[1:]

	biggest_value = 0.0

	for line in lines:
		vals = map(float, line.split(","))
		for val in vals:
			if abs(val) > biggest_value:
				biggest_value = abs(val)
	#biggest value is now set

	output = open("data_just_1h_scaled.csv", 'w')

	output.write(first_line)
	output.write("\n")

	for line in lines:
		vals = map(float, line.split(","))
		first = True
		for val in vals:
			if first:
				first = False
			else:
				output.write(",")
			output.write(str(val / biggest_value))
		output.write("\n")



if __name__ == '__main__':
	main()



