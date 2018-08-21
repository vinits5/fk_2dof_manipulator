import csv

samples = 4
data = []

with open('fk_data_ip.csv','r') as csvfile:
	csvreader = csv.reader(csvfile)
	count = 0
	for row in csvreader:
		if count<samples:
			data.append(row)
			count += 1

data = [[float(i) for i in j]for j in data]

print(data)