#!/usr/bin/python2
# keys at minute intervals
class datapoint:
    def __init__(self):
        self.data = { 
                0:0,
                5:0,
                10:0,
                15:0,
                30:0,
                }   

map_of_datapoints ={}

import csv
# either make a directory walker or modify the filename 
OFFSET_FOR_DOLLAR_VALUE = 3
with open('bitcoin.txt', 'rb') as csvfile:
    rdr = reversed(list(csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)))
    count = 0
    for i in xrange(-6,0):
	map_of_datapoints[i]=datapoint()
    for row in rdr:
	map_of_datapoints[count] = datapoint()
        map_of_datapoints[count].data[0] = row[OFFSET_FOR_DOLLAR_VALUE]
        map_of_datapoints[count].data[5] = map_of_datapoints[count -1].data[0]
        map_of_datapoints[count].data[10] = map_of_datapoints[count -2].data[0]
        map_of_datapoints[count].data[15] = map_of_datapoints[count -3].data[0]
        map_of_datapoints[count].data[30] = map_of_datapoints[count -6].data[0]
        a = map_of_datapoints[count]
        printstring = row[0]+"|"+row[3]+"|"
        is_trending = 1
        if a.data[0] > a.data[5]:
            printstring += "1|" 
        else:
            printstring += "0|"
            is_trending = 0
        if a.data[0] > a.data[10]:
            printstring += "1|"
        else:
            printstring += "0|"
            is_trending = 0
        if a.data[0] > a.data[15]:
            printstring += "1|"
        else:
            printstring += "0|"
            is_trending = 0
        if a.data[0] > a.data[30]:
            printstring += "1|"
        else:
            printstring += "0|"
            is_trending = 0
# Uncomment the following print statements for debugging
#        print printstring
#        print a.data
        print is_trending
        
        count += 1


