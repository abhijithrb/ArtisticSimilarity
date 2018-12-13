"""
	This script contains the class required for fetching the images and labels for training from the csv or text file
	
	---
	Part of Artistic Similarity Project for the 
	semester project of ECGR 6090 Deep Learning in Computer Vision
	Abhijith Bagepalli
	UNC Charlotte
	Dec '18
"""
import tensorflow as tf
import csv
import random
import numpy as np
import pickle

class getData():
	"""
		Class that reads image names and labels from the given csv or text file.
		The given csv file contains all the 79433 images and labels. 
	"""
	def __init__(self):
		"""
			Constructor method for the class which initialises variables used by the
			other functions of the class.
		"""	
		self.data_file_text = "data_file_text.txt"
		self.classess = []
		self.artist_name = []
		self.file_names = []
		self.sub_file_names = []
		self.sub_classess = []
		self.data = []
		self.labels = []

		self.artist_dict = dict()

		self._count = 0
		self._line_count = 0

	def write_into_file(self, file_names, labels):
		"""
			Writes image filenames and labels into a text file.

			Args:
				file_names: A python list of image filenames.
				labels: A python list of labels.
		"""
		with open(self.data_file_text, "w") as _file:
			for _idx in range(len(file_names)):				
				_file.write("%s %s\n" % (file_names[_idx], labels[_idx]))
			
	def read_from_file(self, _data_file_text):
		"""
			Reads pre-fetched image names and labels from a text file.

			Args:
				_data_file_text: Name of the text file
			Returns:
				data: A python list of image names.
				labels: A python list of labels.
		"""
		with open(_data_file_text, "r", encoding='utf-8') as _file:
			for _line in _file:
				_sp = (_line.rstrip()).split(" ")
				 
				self.data.append(_sp[0])
				self.labels.append(_sp[1])

		return self.data, self.labels

	def read_csv(self, csv_path, _data_dir, num_of_artists, num_of_images_per_artist, write_into_text = True):
		"""
			Fetches image names and labels from the given csv file.

			Args:
				csv_path: Location of the csv file
				_data_dir: Directory where the training images are present
				num_of_artists: Number of artists whose images should be used for training
								If num_of_artists = 40 then the top 40 artists with the most paintings in the dataset will be choosen
				num_of_images_per_artist: Number of images by each artist that should be used for training. If num_of_images_per_artist is 50
										  then 50 random images by each artist will be chosen.

				write_into_text: bool indicating if the filenames and labels read from csv should be saved to a text file.
			Returns:
				data: A python list of image names.
				labels: A python list of labels.
		"""
		
		# Read and store rows from the csv file
		with open(csv_path, encoding='utf-8') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			for row in csv_reader:
				if (self._line_count == 0):
					self._line_count += 1
				else:
					self.file_names.append(row[0])
					self.artist_name.append(row[1])

					if (len(self.classess) == 0):
						self.classess.append(row[1])
					else:
						if row[1] not in self.classess:
							self.classess.append(row[1])					

		# Group paintings to their corresponding artist
		for idx in range(len(self.classess)):		
			for idx2 in range(len(self.artist_name)):
				if (self.classess[idx] == self.artist_name[idx2]):
					if (self.classess[idx] in self.artist_dict):
						self.artist_dict[self.classess[idx]].append(self.file_names[idx2]) 
					else:
						self.artist_dict[self.classess[idx]] = [self.file_names[idx2]]

					
		# Sort the artists by the number of paintings and choose the top Args['num_of_artists'] artists.
		for k in sorted(self.artist_dict, key=lambda k: len(self.artist_dict[k]), reverse=True):
			if self._count < num_of_artists:
				_temp = random.sample(self.artist_dict[k], num_of_images_per_artist)
				for idx in _temp:
					_imagePath = _data_dir + idx
					self.sub_file_names.append(_imagePath)
					self.sub_classess.append(k)

				self._count += 1

		# Randomly select Args['num_of_images_per_artist'] paintings from each artist
		_foo = list(zip(self.sub_file_names, self.sub_classess))
		random.shuffle(_foo)
		self.data, self.labels = zip(*_foo)

		# Save the image names and labels to text file
		if write_into_text:
			self.write_into_file(self.data, self.labels)

		return self.data, self.labels	