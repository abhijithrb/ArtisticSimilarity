"""
	About:
		This script runs the Siamese network on the test pairs.
		
		The test images are given in a csv file whose path is set with 'data_file_path'
		
		The csv file contains 1048576 rows with 4 columns: index, image1, image2, sameArtist. For each of 1048576 test pairs, the csv contains the pair index
		or count, filename for image 1 of the pair, filename for image 2 and their actual label(1 for same artist, 0 otherwise)
		
		The directory for the test images should be with the flag 'DATA_DIR'

		The directory for the checkpints that are to be restored can be set with the flag 'checkpt_dir' 

	Dependent scripts:
		model.py

	Running instruction:
		If setting the flags in the script:
			python test.py

		If setting the flags through terminal/command promp:
			(example on how to set the flags. Add/remove flags as needed)
			python test.py --data_file_path C:/Users/arbag/Documents/DeepLearning_in_CV/Project/Data/Data/submission_info.csv 

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

tf.app.flags.DEFINE_string('device_id', '/gpu:0',
                           'What processing unit to execute on')

tf.app.flags.DEFINE_string('data_file_path', "C:/Users/arbag/Documents/DeepLearning_in_CV/Project/Data/Data/submission_info.csv/submission_info.csv",
                           'Path where the csv/text file containing the file name and labels')

tf.app.flags.DEFINE_string('DATA_DIR', 'C:/Users/arbag/Documents/DeepLearning_in_CV/Project/Data/Data/test/',
                           'Directory where are the images are located')

tf.app.flags.DEFINE_string('checkpt_dir', 'Checkpoints', 'Directory where checkpoints are to be saved')

FLAGS = tf.app.flags.FLAGS

# Size of the image used in the model
model_image_width = 256
model_image_height = 256

# Object for class that creates the network. Defined in Model/model.py 
model_obj = model()

def _init_fn():
	"""
		Creates variables for initializers
	"""
	_init = tf.global_variables_initializer()

	# For GPU usage
	_config = tf.ConfigProto(allow_soft_placement = True)
	_config.gpu_options.allow_growth = True
	_config.gpu_options.allocator_type = 'BFC'
	_config.gpu_options.per_process_gpu_memory_fraction = 0.90

	return _config, _init

def get_test_images(filename):
	"""
		Reads csv file with that contains the filenames and labels
		for test images
	"""
	image_1 = []
	image_2 = []
	label = []

	line_count = 0

	with open(filename, encoding='utf-8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')

		for row in csv_reader:
			if (line_count == 0):
				line_count += 1
			else:
				image_1.append(row[1])
				image_2.append(row[2])
				label.append(int(row[3]))

	return image_1, image_2, label		

def read_image(filename_1, filename_2):
	"""
		Reads the image filenames and stores the image data in Tensors. Also reshapes
		the images to the size required by the model
	"""

	image_string_1 = tf.read_file(filename_1)
	image_decoded_1 = tf.image.decode_jpeg(image_string_1, channels = 3)

	image_string_2 = tf.read_file(filename_2)
	image_decoded_2 = tf.image.decode_jpeg(image_string_2, channels = 3)

	image_1 = tf.image.convert_image_dtype(image_decoded_1, tf.float32)
	image_2 = tf.image.convert_image_dtype(image_decoded_2, tf.float32)

	image_1_resized = tf.image.resize_image_with_pad(image_1, model_image_height, model_image_width, method = tf.image.ResizeMethod.BILINEAR)
	image_2_resized = tf.image.resize_image_with_pad(image_2, model_image_height, model_image_width, method = tf.image.ResizeMethod.BILINEAR)

	image_1_reshaped = tf.reshape(image_1_resized, [1, model_image_width, model_image_height, 3])
	image_2_reshaped = tf.reshape(image_2_resized, [1, model_image_width, model_image_height, 3])

	return image_1_reshaped, image_2_reshaped

def inference_fn(filename_1, filename_2):
	"""
		Finds the prediction of artist similarity on the pair of images
		1 if they are by the same artist, 0 otherwise
	"""

	file_1_path = FLAGS.DATA_DIR + filename_1
	file_2_path = FLAGS.DATA_DIR + filename_2

	input_1, input_2 = read_image(file_1_path, file_2_path) # Function call to the read the images

	logits_1 = model_obj.build_model(input_1, reuse = False,  _isTrain = True) # Passes the 1st image through the model
	logits_2 = model_obj.build_model(input_2, reuse = True,  _isTrain = True) # Passes the 2nd image through the model

	predict_op = model_obj.predict_fn(logits_1, logits_2) # Gets Prediction if two images are by the same artist

	return predict_op


def main(argv=None):

	filename_1 = [] # List to store names of the first image in the pair
	filename_2 = [] # List to store names of the second image in the pair
	label = [] # List to store the actual labels of the pairs
	predicted_labels = [] # List to store the predicted labels of the pairs

	# Placeholder for the image pairs
	image1_placeholder = tf.placeholder(tf.string) 
	image2_placeholder = tf.placeholder(tf.string)
	
	# Function call to read csv file containg the filenames and labels for the image pairs
	filename_1, filename_2, label = get_test_images(FLAGS.data_file_path) 

	# Function call to pass the pair of images through the model to get prediction on artistic similaritt
	predict_op = inference_fn(image1_placeholder, image2_placeholder)

	# Function call to get the variables for TF initializers
	config, _init = _init_fn()

	# TF ops for restoring checkpoint of trained model
	saver = tf.train.Saver()

	# Name and path for the checkpoints to restore
	CHECKPOINTS_PATH = os.path.join(FLAGS.checkpt_dir, 'model')
	
	with tf.Session(config = config) as sess:
		
		# Restore the checkpoints of the trained model
		saver.restore(sess, CHECKPOINTS_PATH)

		# Run for all the image pairs in the test list
		for idx in range(len(label)):

			# TF initializers
			sess.run(_init)
			
			# Runs the op to execute the model
			prediction = sess.run(predict_op, feed_dict = {image1_placeholder: filename_1[idx], image2_placeholder: filename_2[idx]})

			# Stores the predictions of each image pair
			if( tf.cast(prediction, tf.int32).eval() == 0):
				predicted_labels.append(0)

			else:
				predicted_labels.append(1)
			
	# Compares the predicted labeles with the actual labels and finds the accuracy 	
	comparisons = sum(x == y for x,y in zip(predicted_labels, label))
	accuracy = comparisons / len(label)
	
	print('accuracy: ', accuracy)


if __name__ == '__main__':
    tf.app.run()