good_data = []
good_data_file = open("good_data.txt")

for line in good_data_file:
	good_data.append("data/5-minute/" + line[:-1] + ".txt")

#good_data now contains the names to all of the files with complete data