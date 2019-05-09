"""
	About:
		Script that trains a Siamese Network for a pair of images to determine if the two paintings are by the same artist.
		
		The network uses a custom 12 layer archtecture followed by 2 fully connected layers. The last layer represents the high dimension feature
		vector of an image. The network takes as its input a 256x256 RGB image.

		Two images are passed through the same network with same weights and the final layers for the two images are used to compute the constrastive loss.
		The constrastive loss compares the Euclidian distances between the the feature vectors of two images to determine how similar they are.
		For two similar images, the Euclidian distance will be smaller.
		The loss is minimised using a Gradient Descent Optimizer.

		The training data comprises of a total of 79433 images by 1584 different artists. This data is present in Data/train_infocsv/train_info.csv which consits
		of 79433 rows with 6 columns: filename artist title style genre date. Only columns filename and artist were used for this model.

		The flag 'num_of_images_per_artist' specifies how many artists from the csv to consider. For example, if 'num_of_images_per_artist' is set to 50, the top 50
		artists with the most images in the dataset will be choosen for training. 

		The flag 'num_of_images_per_artist' specifies how many images by each artist should be taken for training. If 'num_of_images_per_artist' is set to 25, then
		25 random images by each of 50 artists will be taken for training.

		When the flag 'read_prefetched_Data' is set to False, for the above example, the script will read the csv files and select 25 random images from the top 50
		artists along with their labels. These filenames and labels can be stored in a text file by setting the flag 'saveData_to_file' to True. It will store the 
		1250 image filenames and corresponding labels.

		When the flag 'read_prefetched_Data' is set to True, the script will fetch the 1250 image filenames and corresponding labels saved previously.
		During developing stage, when we want to compare a model's performance against different hyper parameters/architectures, it is useful to train/test on the
		same set of images. It is for this reason I have the option of writing the filenames and labels to text file so that the same images can be used since when
		reading from the csv file, I shuffle and select different images by the artists each time. The csv script can be used for the final training.

		The path to the csv or the text file can be set with the flag 'data_file_path'. Note: the text file is saved in the same directory as this script. The path
		is defined in the getData.py script

		The directory where the images lie is set using the flag 'DATA_DIR'
	
	Dependent scripts:
		BatchGenerator.py, getData.py, model.py

	Running instruction:
		If setting the flags in the script:
			python train.py

		If setting the flags through terminal/command promp:
			(example on how to set the flags. Add/remove flags as needed)
			python train.py --num_of_artists 100 --num_of_images_per_artist 50 --read_prefetched_Data False
	
	---
	Part of Artistic Similarity Project for the 
	semester project of ECGR 6090 Deep Learning in Computer Vision
	Abhijith Bagepalli
	UNC Charlotte
	Dec '18
"""
import tensorflow as tf
from BatchGenerator import *
from model import *
import time

# Args to the script:
tf.app.flags.DEFINE_string('device_id', '/gpu:0',
                           'What processing unit to execute on')

tf.app.flags.DEFINE_integer('num_of_artists', 5, 
						   'Number of artists to load from the entire dataset for training. Entire dataset contains 1584 different artists')

tf.app.flags.DEFINE_integer('num_of_images_per_artist', 5, 
						   'Number of paintings per artist to load for training')

tf.app.flags.DEFINE_boolean('read_prefetched_Data', False,
                           'Flag to indicate whether to read pre-fetched file names and labels from text file. If false, will fetch \
                           random (FLAGS.num_of_images_per_artist) paintings by the top (FLAGS.num_of_artists) artists from the original csv.\
                           Once read from csv file, the same image names and labels can be saved in a text file the next time.')

tf.app.flags.DEFINE_boolean('saveData_to_file', False,
                           'Flag to indicate whether to save new file names and labels which were read from the csv to a text file. \
                           This was added so that during developing stage, training and comparison of models can be done on the same set of images \
                           since the same set of images are not always read from the original csv file.')

tf.app.flags.DEFINE_string('data_file_path', "C:/Users/arbag/Documents/DeepLearning_in_CV/Project/Data/Data/train_infocsv/train_info.csv",
                           'Path where the csv/text file containing the training filenames and labels')#data_file_text.txt

tf.app.flags.DEFINE_string('DATA_DIR', 'C:/Users/arbag/Documents/DeepLearning_in_CV/Project/Data/Data/training_images/',
                           'Directory where the images are located')

tf.app.flags.DEFINE_integer('BATCH_SIZE', 1, 'Batch size')

tf.app.flags.DEFINE_integer('EPOCHS', 3,
                           'Number of epochs')

tf.app.flags.DEFINE_float('LEARNING_RATE', 0.0001,
                           'Learning Rate for Gradient Descent Algorithm')

tf.app.flags.DEFINE_string('summaries_dir', 'Log_files', 'Sub Directory for storing summaries')

tf.app.flags.DEFINE_string('checkpt_dir', 'Checkpoints', 'Sub Directory where checkpoints are to be saved')

FLAGS = tf.app.flags.FLAGS

# A dictionary containing variables required by the BatchGenerator class for reading the images
params = {}
	
params['data_file_path'] = FLAGS.data_file_path
params['num_of_artists'] = FLAGS.num_of_artists
params['num_of_images_per_artist'] = FLAGS.num_of_images_per_artist
params['saveData_to_file'] = FLAGS.saveData_to_file
params['read_prefetched_Data'] = FLAGS.read_prefetched_Data
params['batch_size'] = FLAGS.BATCH_SIZE
params['data_dir'] = FLAGS.DATA_DIR

# Object for BatchGenerator class which handels the data generation pipeline. Defined in BatchGenerator.py
batch_obj = BatchGenerator(params)

# Object for the class, model, which defines and creates the CNN. Defined in model.py 
model_obj = model()

def _init_fn(data_itr_train, data_itr_valid):
	"""
		Creates variables for initializers
		Args:
			data_itr_train: Initializer for iterator for tf.data.Dataset for training data
			data_itr_valid: Initializer for iterator for tf.data.Dataset for validation data

		Returns:
			_config: A tf.ConfigProto for setting GPU parameters
			_init: A dictionary containing the TF initializer and the iterators for the training and validation data
	"""
	_init = {}
	
	_init['tf_varaiables'] = tf.global_variables_initializer() # TF initializer
	
	# Initializers for tf.data.Dataset
	_init['train_data_iterator'] = data_itr_train
	_init['valid_data_iterator'] = data_itr_valid

	# For GPU usage
	_config = tf.ConfigProto(allow_soft_placement = True)
	_config.gpu_options.allow_growth = True
	_config.gpu_options.allocator_type = 'BFC'
	_config.gpu_options.per_process_gpu_memory_fraction = 0.90

	return _config, _init

def train_fn(input_1, input_2, Labels):
	"""
		Function that trains the network on pairs of images. 
		Args:
			input_1: A tensor containing the data for image 1 of the siamese network. Has shape of [batch_size 256 256 3]
			input_2: A tensor containing the data for image 2 of the siamese network. Has shape of [batch_size 256 256 3]
			Labels: A tensor containing the labels for the image pair passed. 

		Returns:
			train_op: Training op. Runs the optimizer function on the loss.
			loss_op: Loss op. Computes constrastive loss. Used for printing the loss in main().
			accuracy: Finds the accuracy of predictions on a batch. Used for printing the accuracy in main().

	"""

	# Passes a pair of images to the network and extracts their feature vectors
	logits_1 = model_obj.build_model(input = input_1, reuse = False, _isTrain = True)
	logits_2 = model_obj.build_model(input = input_2, reuse = True, _isTrain = True)

	# Calculates the constrastive loss on the feature vectors on a pair of images
	loss_op, predict_op = model_obj.calculate_loss(logits_1, logits_2, Labels, margin = 0.5)

	# Optimizes the loss - Gradient Descent Optimizer is used
	train_op = model_obj.optimizer(loss_op, FLAGS.LEARNING_RATE)	

	# Finds the accuracy of the model
	accuracy = tf.reduce_mean(tf.cast(tf.equal(Labels, predict_op), tf.float32))

	return train_op, loss_op, accuracy

def validation_fn(input_1, input_2, Labels):
	"""
		Function that tests the model on the validation data 

		Args:
			input_1: A tensor containing the validation data for image 1 of the siamese network. Has shape of [batch_size 256 256 3]
			input_2: A tensor containing the validation data for image 2 of the siamese network. Has shape of [batch_size 256 256 3]
			Labels: A tensor containing the labels for the image pair passed. 
		Returns:
			loss_op: Loss op. Computes constrastive loss. Used for printing the loss in main().
			accuracy: Finds the accuracy of predictions on a batch. Used for printing the accuracy in main().
	"""

	# Passes a pair of images to the network and extracts their feature vectors for the validation data
	logits_1 = model_obj.build_model(input = input_1, reuse = True, _isTrain = False)
	logits_2 = model_obj.build_model(input = input_2, reuse = True, _isTrain = False)

	# Calculates the constrastive loss on the feature vectors of a pair of images for the validation data
	loss_op, predict_op = model_obj.calculate_loss(logits_1, logits_2, Labels, margin = 0.5)

	# Finds the accuracy of the model for the validation data
	accuracy = tf.reduce_mean(tf.cast(tf.equal(Labels, predict_op), tf.float32))

	return loss_op, accuracy


def main(argv = None):
	
	# Calls the get_next_batch function of the BatchGenerator class which fetches the data in batches for training
	input_1, input_2, Labels, data_params = batch_obj.get_next_batch()	

	# Function call to train the network on a pair of images
	train_op, train_loss_op, train_accuracy_op = train_fn(input_1['train'], input_2['train'], Labels['train'])
	
	# Function call to test the network on validation data
	valid_loss_op, valid_accuracy_op = validation_fn(input_1['valid'], input_2['valid'], Labels['valid'])

	# Function call that gets initializer variables
	config, _init = _init_fn(data_params['train_iterator_init'], data_params['valid_iterator_init'])

	# Summaries to store for Tensorboard
	train_loss_summary = tf.summary.scalar(name='train_loss_op', tensor=train_loss_op)
	valid_loss_op_summary = tf.summary.scalar(name='valid_loss_op', tensor=valid_loss_op)	
	merged = tf.summary.merge_all()	
	
	# Summaries to monitor validation results
	valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/valid_writer') 

	# TF ops for saving checkpoint of trained model
	saver = tf.train.Saver()

	# Variable to store best accuracy
	best_accuracy = 0

	with tf.Session(config = config) as sess:		
		
		# TF initializers
		sess.run(_init)	

		# Summaries to monitor training results
		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
		
		for epoch_index in range(FLAGS.EPOCHS):

			# Variables to store accuracies for each epoch
			train_accuracy = 0
			valid_accuracy = 0
						
			start_time = time.time() # For timing analysis
			
			# Print loss and accuracy once in 10 epochs or if final epoch
			if epoch_index % 10 == 0 or ( (epoch_index + 1) == FLAGS.EPOCHS):

				# Run for all batches in the training set
				for batch_index in range(data_params['num_of_train_iterations']):
		
					# Runs the op for taining the model and getting the loss and accuracy
					summary, _, train_loss_value, train_batch_accuracy = sess.run([merged, train_op, train_loss_op, train_accuracy_op])
					
					# Adds up the accuracies of each batch
					train_accuracy += train_batch_accuracy

					# For Tensorboard
					train_writer.add_summary(summary, epoch_index)					

				# Finds the accuracy accross the entire training set
				final_train_accuracy = train_accuracy / data_params['num_of_train_iterations']

				print ('epoch: ', epoch_index)
				print ('Train_loss: ', train_loss_value)
				print ('train_accuracy: ', final_train_accuracy)

				# Run for all batches in the validation set
				for batch_index in range(data_params['num_of_valid_iterations']):
					
					# Runs the op for getting the loss and accuracy of the validation set
					valid_loss_value, valid_batch_accuracy = sess.run([valid_loss_op, valid_accuracy_op])
					
					# Adds up the accuracies of each batch
					valid_accuracy += valid_batch_accuracy

					# For Tensorboard
					valid_writer.add_summary(summary, epoch_index)

				# Finds the accuracy accross the entire validation set
				final_valid_accuracy = valid_accuracy / data_params['num_of_valid_iterations']

				print ('Valid_loss: ', valid_loss_value)
				print ('Valid_accuracy: ', final_valid_accuracy)

				# Checks if current validation accuracy is greater than the best accuracy
				# If ture, Saves checkpoints corresponding to the weights that achieved the best performance on the validation set 
				if (final_valid_accuracy >= best_accuracy):
					
					# Name and path for the checkpoints
					CHECKPOINTS_PATH = os.path.join(FLAGS.checkpt_dir, 'model')

					# TF op to save the checkpoints
					saver.save(sess, CHECKPOINTS_PATH)

					# Updates the overall best accuracy
					best_accuracy = final_valid_accuracy

			# Perform training without printing the losses and accuracies
			else:
				
				# Run for all batches in the training set
				for batch_index in range(data_params['num_of_train_iterations']):
		
					# Runs op to train the network
					summary, _ = sess.run([merged, train_op])
					train_writer.add_summary(summary, epoch_index)
			
			# For calculation time elapsed
			elapsed_time = time.time() - start_time
			print('Time taken for epoch', str(epoch_index), ': ', str(elapsed_time))
		

if __name__ == '__main__':
    tf.app.run()
