import tensorflow as tf 
import os
import numpy as np
from neural_network.neural_network import neural_network
from neural_network.network_structure.Logger import Logger
from IPython import embed
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode',type=str,default='train',help='Mode of operation')
parser.add_argument('--weights',type=str,default='1.ckpt',help='Path of weights')
args = parser.parse_args()

# Create a Neural Network Class.
nn = neural_network()
# Save network structure in logs.
nn.save_network_structure()

L1,L2 = 1,1

def end_effector_pose(t1,t2):
	return [L1*np.cos(t1)+L2*np.cos(t1+t2),L1*np.sin(t1)+L2*np.sin(t1+t2)]

# Create logger file for tensorboard.
# Get the path from neural network class.
logger = Logger(nn.path)
episodes = 5000
batch_size = 100
samples = 10000

data,poses = nn.data(samples)
data = np.asarray(data)
poses = np.asarray(poses)

with tf.Session() as sess:
	# Define Tensors.
	nn.create_model()
	# Initialize tensors.
	nn.session_init(sess)
	if args.mode == 'train':
		steps = 0
		for i in range(episodes):
			Loss = 0
			for j in range(samples/batch_size):
				ip = data[j*batch_size:batch_size*(j+1)]
				op = poses[j*batch_size:batch_size*(j+1)]
				output = nn.forward(data)
				# print(output)
				loss1,loss2,loss = nn.backward(data,poses,batch_size)
				# print('Loss1: {}'.format(loss1))
				# print('Loss2: {}'.format(loss2))
				# print('Loss: {}'.format(loss))
				Loss += loss
				logger.log_scalar(tag='Loss per step',value=loss,step=steps)
				steps += 1
				nn.print_data("Loss per step no",j,loss)
			logger.log_scalar(tag='Average Loss',value=Loss/((i+1)*batch_size),step=i)
			print('\nAverage Loss for episode number {}: {}'.format(i+1,Loss/(batch_size)))
			if (i+1)%10 == 0:
				nn.save_weights(i+1)

		print('Training Result: ')
		# print(nn.forward([[0,0]]))
		print(nn.forward(data))
		print(poses)

	if args.mode == 'test':
		nn.load_weights(args.weights)
		data = [[0,0],[0,np.pi/2],[np.pi/2,0]]
		print(nn.forward(data))
