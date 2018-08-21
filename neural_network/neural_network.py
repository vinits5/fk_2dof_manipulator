import tensorflow as tf 
from network_structure.network_structure import network_structure
import os
import datetime
import shutil
import sys
import csv

class neural_network():
	# Initialize the path for storing data.
	def __init__(self):
		# Class for neural network structure.
		self.ns = network_structure()
		now = datetime.datetime.now()
		path = os.getcwd()
		try:
			os.mkdir('log_data')
		except:
			pass
		self.path = os.path.join(path,'log_data/',now.strftime("%Y-%m-%d-%H-%M-%S"))
		os.mkdir(self.path)

	# Initialize the variables of Neural Network.
	def session_init(self,sess):
		self.sess = sess
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep = 100)

	# Define the network structure.
	def create_model(self):
		self.ns.structure()

	# Forward pass of neural network.
	def forward(self,ip):
		op = self.sess.run([self.ns.output],feed_dict={self.ns.x:ip})
		return op

	# Backpropagation of neural network.
	def backward(self,ip,y,batch):
		l1,l2,l,_ = self.sess.run([self.ns.loss1,self.ns.loss2,self.ns.loss,self.ns.updateModel],feed_dict={self.ns.x:ip,self.ns.y:y,self.ns.batch_size:batch})
		return l1,l2,l		

	# Store weights for further use.
	def save_weights(self,episode):
		path_w = os.path.join(self.path,'weights')
		try:
			os.chdir(path_w)
		except:
			os.mkdir(path_w)
			os.chdir(path_w)
		path_w = os.path.join(path_w,'{}.ckpt'.format(episode))
		self.saver.save(self.sess,path_w)

	def load_weights(self,weights):
		self.saver.restore(self.sess,weights)

	# Store network structure in logs.
	def save_network_structure(self):
		curr_dir = os.getcwd()
		src_path = os.path.join(curr_dir,'neural_network','network_structure','network_structure.py')
		target_path = os.path.join(self.path,'network_structure.py')
		shutil.copy(src_path,target_path)

	def print_data(self,text,step,data):
		text = "\r"+text+" %d: %f"
		sys.stdout.write(text%(step,data))
		sys.stdout.flush()

	def batch_size(self,batch):
		self.sess.run(self.ns.batch_size,feed_dict={self.ns.batch_size:batch})

	def data(self,samples):
		data,poses = [],[]
		with open('fk_data_ip.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			count = 0
			for row in csvreader:
				if count<samples:
					data.append(row)
					count += 1

		with open('fk_data_op.csv','r') as csvfile:
			csvreader = csv.reader(csvfile)
			count = 0
			for row in csvreader:
				if count<samples:
					poses.append(row)
					count += 1
		data = [[float(i) for i in j]for j in data]
		poses = [[float(i) for i in j]for j in poses]
		return data,poses
