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
for i in range(90):
	data.append([0,i*(np.pi/180)])

L1,L2 = 1,1
def end_effector_pose(t1,t2):
	return [round(L1*np.cos(t1)+L2*np.cos(t1+t2),3),round(L1*np.sin(t1)+L2*np.sin(t1+t2),3)]

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
plt.scatter(poses_kinematics_x,poses_kinematics_y,s=20,c='r',)
plt.plot(poses_kinematics_x,poses_kinematics_y,c='r',label='Kinematics Positions')
plt.scatter(poses_nn_x,poses_nn_y,s=20,c='b')
plt.plot(poses_nn_x,poses_nn_y,c='b',label='nn Positions')
plt.grid(True)
ax.legend(loc='best')
plt.show()


