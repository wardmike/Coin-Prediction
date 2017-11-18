if __name__ == '__main__':
	fl = open("../data/condensed/data_just_24h.csv")
	firstLine = ""
	totalCount = -1
	for line in fl:
		if totalCount == -1:
			firstLine = line
		totalCount += 1

	output_90 = open("../data/condensed/data_just_24h_90.csv", 'w')
	output_10 = open("../data/condensed/data_just_24h_10.csv", 'w')
	output_90.write(firstLine)
	output_10.write(firstLine)

	fl = open("../data/condensed/data_just_24h.csv")
	count = -1
	for line in fl:
		if count != -1:
			if count < 0.9 * totalCount:
				output_90.write(line)
			else:
				output_10.write(line)
		count += 1


