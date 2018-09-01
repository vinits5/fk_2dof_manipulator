import csv
import matplotlib.pyplot as plt 
import numpy as np

step,reward = [],[]

# filter_size = 1095

def filter(data):
	return sum(data)/filter_size

val = 1500
time = 3
with open('loss_step.csv','r') as csvfile:
	csvreader = csv.reader(csvfile)
	csvreader.next()
	for row in csvreader:
		if int(row[1])<15000:
			step.append(int(row[1]))
			reward.append(float(row[2]))

def filtration(reward,step,filter_s):
	global filter_size
	filter_size = filter_s
	reward = reward[0:filter_size/2]+reward+reward[len(reward)-filter_size/2:len(reward)]
	for i in range(filter_size/2,len(reward)-filter_size/2-1):
		data = reward[i-filter_size/2:i+filter_size/2+1]
		reward[i]=filter(data)

	reward = reward[filter_size/2:len(reward)-filter_size/2]
	# print len(reward)
	reward = reward[0:len(reward)-1]
	step = step[0:len(step)-1]
	return reward,step


# val = 650
import sys
val = int(sys.argv[2])
time = float(sys.argv[1])
for i in range(val,val+1):
	plt.clf()
	r,s = filtration(reward,step,i)
	# print len(r)
	# print len(s)
	plt.plot(s,r,c='b',linewidth=1)
	plt.scatter(s,r,s = 8)
	# plt.text(200,200,'Filter Size: '+str(i))
	# plt.axis([0,9000,-300,500])
	plt.xlabel('Steps')
	plt.ylabel('Loss')
	plt.title('Loss per Step')
	plt.pause(time)
