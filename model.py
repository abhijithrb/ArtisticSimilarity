"""
	This script contains the class which defines the architecture of the network.
	The network uses a custom 12 layer cnn followed by 2 fully connected layers.
	The last last layer for each image acts as the feature vector on which the constrastive loss
	is found.
	
	---
	Part of Artistic Similarity Project for the 
	semester project of ECGR 6090 Deep Learning in Computer Vision
	Abhijith Bagepalli
	UNC Charlotte
	Dec '18
"""
import tensorflow as tf 
import numpy as np  
import os

class model():
	"""
		Class that defines and creates the CNN model used
	"""
	def __init__(self):
		"""
			Constructor method of the class that initializes variables used for weights and biases of the CNN
		"""
		self.kernel_init = tf.initializers.random_normal(mean = 0, stddev = 10**-2)
		self.bias_init = tf.initializers.random_normal(mean = 0.5, stddev = 10**-2)

	def build_model(self, input, reuse, _isTrain = False):
		"""
			Creates the architecture for the CNN network. The network consists of 12 conv layers followed by 2 fully connected layers 
			Args:
				input: Input image to the network. Must have a shape of [batch_size 256 256 3]
				reuse: Flag indicating if layer should be used. True for the second image of a pair. This way the same network is used for both images
				_isTrain: Flag indicting if training or testing
			Returns:
				_logits: The final layer of the network. Represents the high dimension feature vector of an image
		"""
		with tf.name_scope("model"):
			with tf.variable_scope("conv1") as scope:
				conv1 = tf.layers.conv2d(inputs=input, filters = 16, kernel_size = [11, 11], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_1 = tf.layers.batch_normalization(conv1, axis=1, training=_isTrain, name=scope, reuse=reuse)

				pool1 = tf.layers.max_pooling2d(inputs = batch_1, pool_size = [2, 2], strides = 2, padding = 'SAME')

			with tf.variable_scope("conv2") as scope:
				conv2 = tf.layers.conv2d(inputs=pool1, filters = 32, kernel_size = [9, 9], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_2 = tf.layers.batch_normalization(conv2, axis=1, training=_isTrain, name=scope, reuse=reuse)

			with tf.variable_scope("conv3") as scope:
				conv3 = tf.layers.conv2d(inputs=batch_2, filters = 32, kernel_size = [9, 9], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_3 = tf.layers.batch_normalization(conv3, axis=1, training=_isTrain, name=scope, reuse=reuse)

			with tf.variable_scope("conv4") as scope:
				conv4 = tf.layers.conv2d(inputs=batch_3, filters = 32, kernel_size = [9, 9], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_4 = tf.layers.batch_normalization(conv4, axis=1, training=_isTrain, name=scope, reuse=reuse)

				pool2 = tf.layers.max_pooling2d(inputs = batch_4, pool_size = [2, 2], strides = 2, padding = 'SAME')

			with tf.variable_scope("conv5") as scope:
				conv5 = tf.layers.conv2d(inputs=pool2, filters = 64, kernel_size = [7, 7], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_5 = tf.layers.batch_normalization(conv5, axis=1, training=_isTrain, name=scope, reuse=reuse)

			with tf.variable_scope("conv6") as scope:
				conv6 = tf.layers.conv2d(inputs=batch_5, filters = 64, kernel_size = [7, 7], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_6 = tf.layers.batch_normalization(conv6, axis=1, training=_isTrain, name=scope, reuse=reuse)

			with tf.variable_scope("conv7") as scope:
				conv7 = tf.layers.conv2d(inputs=batch_6, filters = 64, kernel_size = [7, 7], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)
				batch_7 = tf.layers.batch_normalization(conv7, axis=1, training=_isTrain, name=scope, reuse=reuse)

				pool3 = tf.layers.max_pooling2d(inputs = batch_7, pool_size = [2, 2], strides = 2, padding = 'SAME')

			with tf.variable_scope("conv8") as scope:
				conv8 = tf.layers.conv2d(inputs = pool3, filters = 128, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)

				batch_8 = tf.layers.batch_normalization(conv8, axis=1, training=_isTrain, name=scope, reuse=reuse)

			with tf.variable_scope("conv9") as scope:
				conv9 = tf.layers.conv2d(inputs = batch_8, filters = 128, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)

				batch_9 = tf.layers.batch_normalization(conv9, axis=1, training=_isTrain, name=scope, reuse=reuse)

			with tf.variable_scope("conv10") as scope:
				conv10 = tf.layers.conv2d(inputs = batch_9, filters = 128, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)

				batch_10 = tf.layers.batch_normalization(conv10, axis=1, training=_isTrain, name=scope, reuse=reuse)

				pool4 = tf.layers.max_pooling2d(inputs = batch_10, pool_size = [2, 2], strides = 2, padding = 'SAME')

			with tf.variable_scope("conv11") as scope:
				conv11 = tf.layers.conv2d(inputs = pool4, filters = 256, kernel_size = [4, 4], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)

				batch_11 = tf.layers.batch_normalization(conv11, axis=1, training=_isTrain, name=scope, reuse=reuse)

				pool5 = tf.layers.max_pooling2d(inputs = batch_11, pool_size = [2, 2], strides = 2, padding = 'SAME')

			with tf.variable_scope("conv12") as scope:
				conv12 = tf.layers.conv2d(inputs = pool5, filters = 256, kernel_size = [4, 4], padding = 'same', activation = tf.nn.relu, 
										kernel_initializer = self.kernel_init, bias_initializer = self.bias_init, name = scope, reuse = reuse)

				batch_12 = tf.layers.batch_normalization(conv12, axis=1, training=_isTrain, name=scope, reuse=reuse)

				pool6 = tf.layers.max_pooling2d(inputs = batch_12, pool_size = [2, 2], strides = 2, padding = 'SAME')

			with tf.variable_scope("flatt") as scope:
				_flat = tf.layers.flatten(pool6, name = scope)
				
			with tf.variable_scope("dense1") as scope:
				_logits = tf.layers.dense(_flat, 4096, activation = tf.nn.relu, name = scope, reuse = reuse)					
				
		return _logits

	def calculate_loss(self, logits_1, logits_2, label, margin = 0.5):
		"""
			Function that defines the constrastive loss used for the Siamese Network
			Args:
				logits_1: Final layer of the model for image 1 of a pair of images
				logits_2: Final layer of the model for image 2 of a pair of images
				label: Label for the pair of images. 1 if they are by the same artist else 0
				margin: Margin parameter used in the constrastive loss
			Returns:
				loss_op: The constrastive loss calculated for a pair of images
				_prediction: Prediction by the network
		"""
		# Define loss function
		with tf.variable_scope("loss_function"):		

			# Euclidean distance squared
			eucd2 = tf.pow(tf.subtract(logits_1, logits_2), 2)
			
			eucd2 = tf.reduce_mean(eucd2, 1)
			
			# Euclidean distance
			# I add a small value 1e-6 to increase the stability of calculating the gradients for sqrt
			eucd = tf.sqrt(eucd2 + 1e-6)

			# Loss function
			loss_pos = tf.multiply(label, eucd2)
			loss_neg = tf.multiply(tf.subtract(1.0, label), tf.pow(tf.maximum(tf.subtract(margin, eucd), 0), 2))
			loss_op = tf.reduce_mean(tf.add(loss_neg, loss_pos), name = 'constrastive_loss')	

			# Gets the prediction 
			_prediction = self.predict_fn(logits_1, logits_2)					
			
		return loss_op, _prediction

	def optimizer(self, loss, _learning_rate):
		"""
			Runs the Gradient Descent Optimizer on the computed loss.
			Args:
				loss: The calculated constrastive loss
				_learning_rate: Learning rate used by the Gradient Descent Optimizer
			Returns:
				optimizer_op: The op used to execute the optimizer
		"""
		with tf.variable_scope("optimizer_function"):
			# Initialize optimizer
			optimizer_op = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)

		return optimizer_op

	def predict_fn(self, logits_1, logits_2):
		"""
			Gets the prediction for a pair of images.
			If two images are similar, then the Euclidian distance of their feature vectors will be small
			Args:
				logits_1: The logits layer for image 1 in a pair. Represents the high dimension feature vector
				logits_2: The logits layer for image 2 in a pair. Represents the high dimension feature vector
		"""

		with tf.variable_scope("predict"):
			
			# Euclidean distance squared
			eucd2 = tf.pow(tf.subtract(logits_1, logits_2), 2)
			
			eucd2 = tf.reduce_mean(eucd2, 1)
			
			# Euclidean distance
			# I add a small value 1e-6 to increase the stability of calculating the gradients for sqrt
			eucd_predict1 = tf.sqrt(eucd2 + 1e-6)

			condition = tf.less(eucd_predict1, 0.3) # Threshold to compare the Euclidian distance

			res = tf.where(condition, tf.ones_like(eucd_predict1), tf.zeros_like(eucd_predict1))			

		return res