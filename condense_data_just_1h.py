
class TimePoint(object):
	def __init__(self, time):
		self.time = time
		self.data = []

	def add(self, data_point):
		self.data.append(data_point)

	def print_line(self):
		line = ""
		first = True
		for i in self.data:
			if first:
				first = False
			else:
				line += ","
			line += i
		return line


class PriceDatabase(object):

	def __init__(self):
		good_data = []
		good_data_file = open("good_data.txt")
		self.times = []
		self.currencies = []

		for line in good_data_file:
			good_data.append("data/5-minute/" + line[:-1] + ".txt")
		first = True
		for fl in good_data:
			if first:
				self.add_first_currency(fl)
				first = False
			else:
				self.add_currency(fl)


	def add_first_currency(self, filename):
		fl = open(filename)
		first = True
		i = 0
		for line in fl:
			vals = line.split("|")
			if first:
				self.currencies.append(vals[2])
				first = False
			self.times.append(TimePoint(vals[0]))
			self.times[i].add(vals[9])
			i += 1

	def add_currency(self, filename):
		fl = open(filename)
		first = True
		i = 0
		for line in fl:
			vals = line.split("|")
			if first:
				self.currencies.append(vals[2])
				first = False
			self.times[i].add(vals[9])
			i += 1

	def print_to_file(self, filename):
		fl = open(filename, 'w')
		first = True
		currList = ""
		for curr in self.currencies:
			if first:
				first = False
			else:
				currList += ","
			currList += curr
		fl.write(currList)
		fl.write("\n")

		for time in self.times:
			fl.write(time.print_line())
			fl.write("\n")


if __name__ == '__main__':
	data = PriceDatabase()
	data.print_to_file("data_just_1h.csv")
