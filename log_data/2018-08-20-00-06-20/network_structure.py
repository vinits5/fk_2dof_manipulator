import tensorflow as tf 


class network_structure():
	# Use it to initialize weights.
	def weights(self,x,y):
		weights_dict = {'weights':tf.Variable(tf.random_normal([x,y])),'biases':tf.Variable(tf.random_normal([y]))}
		return weights_dict

	# Define the complete neural network here.
	def structure(self):
		self.x = tf.placeholder(tf.float32,shape=(None,2))
		self.y = tf.placeholder(tf.float32,shape=(None,2))
		self.batch_size = tf.placeholder(tf.float32,shape=None)

		self.nodes_layer1 = 200
		self.hidden_layer1 = self.weights(2,self.nodes_layer1)

		# self.nodes_layer2 = 200
		# self.hidden_layer2 = self.weights(self.nodes_layer1,self.nodes_layer2)

		# self.nodes_layer3 = 200
		# self.hidden_layer3 = self.weights(self.nodes_layer2,self.nodes_layer3)

		self.nodes_output = 2
		self.output_layer = self.weights(self.nodes_layer1,self.nodes_output)

		self.l1 = tf.add(tf.matmul(self.x,self.hidden_layer1['weights']),self.hidden_layer1['biases'])
		self.l1 = tf.nn.relu(self.l1)

		# self.l2 = tf.add(tf.matmul(self.l1,self.hidden_layer2['weights']),self.hidden_layer2['biases'])
		# self.l2 = tf.nn.relu(self.l2)

		# self.l3 = tf.add(tf.matmul(self.l2,self.hidden_layer3['weights']),self.hidden_layer3['biases'])
		# self.l3 = tf.nn.relu(self.l3)

		self.output = tf.add(tf.matmul(self.l1,self.output_layer['weights']),self.output_layer['biases'])

		self.loss1 = tf.square(self.output-self.y)
		self.loss2 = tf.reduce_sum(self.loss1)
		self.loss = tf.divide(self.loss2,self.batch_size)
		self.trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
		self.updateModel = self.trainer.minimize(self.loss)