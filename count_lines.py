import glob
output = open("readme.md", 'w+')
output.write("# CryptoData\n\nTrying to upload cryptocurrency data taken every five minutes to:  \n")
output.write("- run algorithms on past data to optimize future returns  \n")
output.write("- have a small community to share ideas and algorithms  \n")
output.write("## Full Data\n<table>\n")
for filename in glob.glob('data/5-minute/*.txt'):
	fl = open(filename)
	count = 0
	for line in fl:
		count = count + 1
	if count == 11194:
		output.write("<tr><td>" + filename[14:-4] + "</td></tr>\n")
output.write("</table>\n\n")

output.write("## Lines\n<table>\n")
for filename in glob.glob('data/5-minute/*.txt'):
	fl = open(filename)
	count = 0
	for line in fl:
		count = count + 1
	output.write("<tr><td>" + filename[14:-4] + "</td><td>" + str(count) + "</td></tr>\n")
output.write("</table>\n")
