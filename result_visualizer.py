import matplotlib.pyplot as plt 
from neural_network.neural_network import neural_network
import argparse
import tensorflow as tf
import numpy as np 

nn = neural_network()

parser = argparse.ArgumentParser()
parser.add_argument('--weights',type=str,default='200.ckpt',help='Weights for neural network')

args = parser.parse_args()

data = []
for i in range(91):
	data.append([i*(np.pi/180),i*(np.pi/180)])

L1,L2 = 1,1
def end_effector_pose(t1,t2):
	return [round(L1*np.cos(t1)+L2*np.cos(t1+t2),3),round(L1*np.sin(t1)+L2*np.sin(t1+t2),3)]

def pose_joint1(t1):
	return [round(L1*np.cos(t1),3),round(L1*np.sin(t1),3)]

with tf.Session() as sess:
	nn.create_model()
	nn.session_init(sess)
	nn.load_weights(args.weights)
	poses_kinematics,poses_nn = [],[]
	for i in range(len(data)):
		pose = end_effector_pose(data[i][0],data[i][1])
		poses_kinematics.append(pose)
		poses_nn.append(nn.forward([data[i]]))

poses_kinematics_x,poses_kinematics_y = [],[]
for i in range(len(poses_kinematics)):
	poses_kinematics_x.append(poses_kinematics[i][0])
	poses_kinematics_y.append(poses_kinematics[i][1])

poses_nn_x,poses_nn_y = [],[]
for i in range(len(poses_nn)):
	poses_nn_x.append(poses_nn[i][0][0][0])
	poses_nn_y.append(poses_nn[i][0][0][1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('End Effector Positions')
ax.set_xlabel('X-positions')
ax.set_ylabel('Y-positions')
# plt.scatter(poses_kinematics_x,poses_kinematics_y,s=20,c='r',)
# plt.plot(poses_kinematics_x,poses_kinematics_y,c='r',label='Kinematics Positions')
# plt.scatter(poses_nn_x,poses_nn_y,s=20,c='b')
# plt.plot(poses_nn_x,poses_nn_y,c='b',label='nn Positions')

X_1,Y_1,X_2,Y_2 = [],[],[],[]

for i in range(len(poses_nn_x)):
	# print(data[i][0])
	plt.clf()
	x,y = pose_joint1(float(data[i][0]))
	plt.axis([-2.5,2.5,-2.5,2.5])
	# plt.grid(True)
	plt.title('End Effector Positions')
	plt.xlabel('X-positions')
	plt.ylabel('Y-positions')
	# plt.scatter(X_1,Y_1,c='r',s=20)
	plt.scatter(X_2,Y_2,c=[0.5,0,0],s=5)

	plt.plot([0,x,poses_nn_x[i]],[0,y,poses_nn_y[i]],linewidth = 3,c=[153/255.0,1,204/255.0])
	plt.scatter([0],[0],c=[0,0,0],s=80)
	plt.scatter([x,poses_nn_x[i]],[y,poses_nn_y[i]],c='b',s=50)
	X_1.append(x)
	Y_1.append(y)
	X_2.append(poses_nn_x[i])
	Y_2.append(poses_nn_y[i])
	plt.text(-2.2,-1.5,'Theta1: ')
	plt.text(-1.5,-1.5,str(data[i][0]*(180/np.pi)))
	plt.text(-2.2,-1.8,'Theta2: ')
	plt.text(-1.5,-1.8,str(data[i][1]*(180/np.pi)))
	plt.text(1,-1.5,'x_pos: ')
	plt.text(1.5,-1.5,str(poses_nn_x[i]))
	plt.text(1,-1.8,'y_pos: ')
	plt.text(1.5,-1.8,str(poses_nn_y[i]))

	plt.pause(0.1)
	plt.savefig('test'+str(i)+'.jpg')
ax.legend(loc='best')
plt.plot(poses_kinematics_x,poses_kinematics_y,c=[0,1,0])
plt.savefig('test'+str(i+1)+'.jpg')
plt.show()
