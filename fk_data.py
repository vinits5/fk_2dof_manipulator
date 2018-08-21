import numpy as np
import csv

L1,L2 = 1,1
def end_effector_pose(t1,t2):
	return [round(L1*np.cos(t1)+L2*np.cos(t1+t2),3),round(L1*np.sin(t1)+L2*np.sin(t1+t2),3)]

# Data Creation
samples = 100000
data = (np.pi/2)*np.random.random_sample((samples,2))
poses = []
for j in range(data.shape[0]):
	pose = end_effector_pose(data[j][0],data[j][1])
	poses.append(pose)
poses = np.asarray(poses)

with open('fk_data_ip.csv','w') as csvfile:
	csvwriter = csv.writer(csvfile)
	for i in range(data.shape[0]):
		csvwriter.writerow(data[i])
csvfile.close()

with open('fk_data_op.csv','w') as csvfile:
	csvwriter = csv.writer(csvfile)
	for i in range(poses.shape[0]):
		csvwriter.writerow(poses[i])
csvfile.close()