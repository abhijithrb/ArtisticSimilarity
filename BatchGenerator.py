"""
	Script contains class required for creatings equal number of unique same and different pair of images along with their labels.
	The scipt also uses the tf.data.Dataset to create a data pipeline to sent to the network.
	
	---
	Part of Artistic Similarity Project for the 
	semester project of ECGR 6090 Deep Learning in Computer Vision
	Abhijith Bagepalli
	UNC Charlotte
	Dec '18
"""
from itertools import combinations
import tensorflow as tf
import numpy as np
import random
from random import shuffle
from numpy.random import choice
import os
from getData import *

class BatchGenerator():
	"""
		Class that is used to create unique positive and negtive pairs from the training images
		and use the tf.data.Dataset to creat an input data pipeline that is used to feed input data
		into the model.
	"""
	def __init__(self, params):
		"""
			Constructor method of the class which initializes variables and calls function to read image names and labels 
		"""
		self._data_file_path = params['data_file_path']
		self._num_of_artists = params['num_of_artists']
		self._num_of_images_per_artist = params['num_of_images_per_artist']
		self._saveData_to_file = params['saveData_to_file']
		self._read_prefetched_Data = params['read_prefetched_Data']
		self._batch_size = params['batch_size']
		self._data_dir = params['data_dir']

		self._height = 256
		self._width = 256

		self._left_image = {} 
		self._right_image = {} 
		self._labels = {}
		self._params = {}
		self._trainFlag = True

		# Object for the getData class which reads filenames and labels from csv/text file. Defined in the getData.py
		_csv_obj = getData()
		
		# Fetches the training data either from csv or text file. 
		if not self._read_prefetched_Data:
			print('Loading file names and labels from csv.')
			self.data, self.label = _csv_obj.read_csv(self._data_file_path, self._data_dir, self._num_of_artists, self._num_of_images_per_artist, self._saveData_to_file)
		else:
			print('Loading existing file names and labels from text file.')
			self.data, self.label = _csv_obj.read_from_file(self._data_file_path)

		# Splits the training data into training and validation sets (80-20)
		self.train_data = self.data[:int(len(self.data) * 0.8)]
		self.train_labels = self.label[:int(len(self.label) * 0.8)]

		self.valid_data = self.data[int(len(self.data) * 0.8):]
		self.valid_labels = self.label[int(len(self.label) * 0.8):]		
		
	def create_pairs(self, Labels, is_train = True):
		"""
			Function that creates equal number of unique positive and negative pairs
			Args: 
				Labels: A python list of labels for the training data
				is_train: Flag to indiacte if training data or validation data
			Returns:
				similar_pairs: A python list of pairs of indices of images by the same artist
							   These indices indicate the postion of elements in the list self.train_data/self.valid_data
							   Eg: if similar_pairs contains [(1,2), (3,6)], elements 1 & 2 of self.train_data will form one pair, similarly 3 & 6
				disimilar_pairs: A python list of pairs of indices of images by different artists
		"""
		num_idx = dict()
		
		for idx, num in enumerate(Labels):
			if num in num_idx:
				num_idx[num].append(idx)
			else:
				num_idx[num] = [idx]
		
		similar_pairs = []
		all_pairs = []
		index = []
		
		for x,y in num_idx.items():
			index = list(y) + index
			similar_pairs = list(combinations(y, 2)) + similar_pairs
		
		all_pairs = list(combinations(index, 2))
		
		disimilar_pairs_all = list(set(all_pairs).difference(similar_pairs))
						
		disimilar_pairs = random.sample(disimilar_pairs_all, len(similar_pairs))
		
		if is_train:
			print('Number of Positive pairs taken for training: ', len(similar_pairs))
			print('Number of Negative pairs taken for training: ', len(disimilar_pairs))	
		else:
			print('Number of Positive pairs taken for validation: ', len(similar_pairs))
			print('Number of Negative pairs taken for validation: ', len(disimilar_pairs))	

		return similar_pairs, disimilar_pairs

	def get_pairs(self, trainData, Labels, is_train = True):
		"""
			Function that generates pairs of images and their corresponding labels
			Args:
				trainData: A python list containing the image names
				Labels: A python list containing the artist names for the images
				is_train: Flag indicating if training data or validation data is used
			Returns:
				left_data: A python list containing the 1st images of a pair
				right_data: A python list containing thr 2nd images of a pair
				_labels: A python list containing the Labels for the pairs. 1 if a pair is by the same artist, else 0
		"""
		
		# Gets the indices of the images that by the same artist and different artists
		similar_pair_index, disimilar_pair_index = self.create_pairs(Labels, is_train)
			
		similar_pair_data = []
		disimilar_pair_data = []

		left_data = []
		right_data = []
		_labels = []

		# Gets same and different pairs
		for idx in range(len(disimilar_pair_index)):
			similar_pair_data.append((trainData[similar_pair_index[idx][0]], trainData[similar_pair_index[idx][1]]))
			disimilar_pair_data.append((trainData[disimilar_pair_index[idx][0]], trainData[disimilar_pair_index[idx][1]]))
		
		# Shuffles the list to ensure randomness. Otherwise all works by Artist 1, followed by Artist 2 and so on appears.
		random.shuffle(similar_pair_data)
		
		# Stores the pairs and their corresponding labels in a list
		for idx in range(len(similar_pair_data)):
			left_data.append(similar_pair_data[idx][0])
			right_data.append(similar_pair_data[idx][1])		
			_labels.append(1)

			left_data.append(disimilar_pair_data[idx][0])
			right_data.append(disimilar_pair_data[idx][1])		
			_labels.append(0)
		
		return left_data, right_data, _labels

	def parse_function(self, filename_left, filename_right, label):
		"""
			Function that reads the filenames and stores the JPEG data in tensors.
			Also resizes the images to 256x256

			Args:
				filename_left: File name for the 1st image
				filename_right: File name for the 2nd image
				label: Label of the image pair
			Return:
				left_image_resized: The JPEG data of filename_left resized to 256x256
				right_image_resized: The JPEG data of filename_right resized to 256x256
				label: Label of the image pair
		"""
		image_string_left = tf.read_file(filename_left)
		image_decoded_left = tf.image.decode_jpeg(image_string_left, channels = 3)

		image_string_right = tf.read_file(filename_right)
		image_decoded_right = tf.image.decode_jpeg(image_string_right, channels = 3)

		left_image = tf.image.convert_image_dtype(image_decoded_left, tf.float32)
		right_image = tf.image.convert_image_dtype(image_decoded_right, tf.float32)

		left_image_resized = tf.image.resize_image_with_pad(left_image, self._height, self._width, method = tf.image.ResizeMethod.BILINEAR)
		right_image_resized = tf.image.resize_image_with_pad(right_image, self._height, self._width, method = tf.image.ResizeMethod.BILINEAR)

		return left_image_resized, right_image_resized, label

	def get_next_batch(self):
		"""
			Function that gets the pair of images and their labels and uses tf.data.Dataset to create batches and an input pipeline to feed to the
			network
			Returns:
				_left_image: A dictionary containing image 1 of a pair for training and validation sets
				_right_image: A dictionary containing image 2 of a pair for training and validation sets
				_labels: A dictionary containing labels for an image pair for training and validation sets
				_params: A dictionary containing Initializer for the iterator of tf.data.Dataset for training and validation sets and the number of
						iterations needed to go through all batches of the training and validation sets
		"""
		
		# Gets image pairs and their corresponding labels
		_left_train, _right_train, _labels_train = self.get_pairs(self.train_data, self.train_labels, is_train = True)		
		_left_valid, _right_valid, _labels_valid = self.get_pairs(self.valid_data, self.valid_labels, is_train = False)	
		
		# Creates a data pipeline using tf.data.Dataset to create batches to feed to the network
		with tf.name_scope('data_pipeline'):
			_dataset_train = tf.data.Dataset.from_tensor_slices((tf.constant(_left_train), tf.constant(_right_train), tf.constant(_labels_train)))
			_dataset_train = _dataset_train.shuffle(len(self.data))
			_dataset_train = _dataset_train.map(self.parse_function, num_parallel_calls = 4)
			_dataset_train = _dataset_train.repeat()
			_dataset_train = _dataset_train.batch(self._batch_size)
			_dataset_train = _dataset_train.prefetch(1)

			_iterator_train = _dataset_train.make_initializable_iterator()
			_left_train_op, _right_train_op, _label_train_op = _iterator_train.get_next()
			_iterator_init_train_op = _iterator_train.initializer

			_dataset_valid = tf.data.Dataset.from_tensor_slices((tf.constant(_left_valid), tf.constant(_right_valid), tf.constant(_labels_valid)))
			_dataset_valid = _dataset_valid.shuffle(len(self.data))
			_dataset_valid = _dataset_valid.map(self.parse_function, num_parallel_calls = 4)
			_dataset_valid = _dataset_valid.repeat()
			_dataset_valid = _dataset_valid.batch(self._batch_size)
			_dataset_valid = _dataset_valid.prefetch(1)

			_iterator_valid = _dataset_valid.make_initializable_iterator()
			_left_valid_op, _right_valid_op, _label_valid_op = _iterator_valid.get_next()
			_iterator_init_valid_op = _iterator_valid.initializer

			self._left_image['train'] = _left_train_op
			self._left_image['valid'] = _left_valid_op

			self._right_image['train'] = _right_train_op
			self._right_image['valid'] = _right_valid_op
				
			self._labels['train'] = tf.cast(_label_train_op, tf.float32)
			self._labels['valid'] = tf.cast(_label_valid_op, tf.float32)

			self._params['train_iterator_init'] = _iterator_init_train_op
			self._params['valid_iterator_init'] = _iterator_init_valid_op

			self._params['num_of_train_iterations'] = (len(_left_train) + self._batch_size - 1) // self._batch_size
			self._params['num_of_valid_iterations'] = (len(_left_valid) + self._batch_size - 1) // self._batch_size

		return self._left_image, self._right_image, self._labels, self._params