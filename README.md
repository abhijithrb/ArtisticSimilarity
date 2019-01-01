# ArtisticSimilarity
Semester project for ECGR 6090 Deep Learning in Computer Vision at UNC Charlotte.

This project uses a Siamese Network to identify if two paintings are by the same artist. The task was inspired by [this](https://www.kaggle.com/painter-by-numbers) Kaggle challenge.

In a Siamese Architecture, a pair of images is passed to two networks with that share the same architecture and same weights. The output of these two networks are then passed to a contrastive loss function. The contrastive loss basically determines how similar two images are. The loss is calculated using the Euclidian distance. If two images are similar, the Euclidian distance will be small.

The image below shows the graph of the model used for this project:
## Model
![](https://github.com/abhijithrb/ArtisticSimilarity/blob/master/github_images/tensorboard-graph.png)

It is a custom model that consists of 12 conv layers followed by 2 fully connected layers. The last layer of image 1 and image 2 are passed to the contrastive loss function which estimates if the two images are by the same artist or not. You can find more details about the implementation of the network architecture and loss in [model.py script](model.py)

## Preparing the data
I used the dataset provided by Kaggle. It consists of 79433 images by 1584 different artists. The dataset can be downloaded directly from the challenge homepage or using the Kaggle API. The latter is recommended as the dataset is extremely big and will be faster downloading it using the API. 
Please refer to [this](https://github.com/Kaggle/kaggle-api) link to learn how to use the Kaggle API to download datasets.

A train_info.csv file is provided as part of the dataset which consits	of 79433 rows with 6 columns: *filename*, *artist*, *title*, *style*, *genre*, *date*. 
Only columns *filename* and *artist* were used for this model. 

In [train.py](train.py), the flag *num_of_images_per_artist* specifies how many artists from the csv to consider for training. For example, if *num_of_images_per_artist* is set to 50, the top 50	artists with the most images in the dataset will be choosen. The flag *num_of_images_per_artist* specifies how many images by each artist should be taken for training. If *num_of_images_per_artist* is set to 25, then	25 random images by each of 50 artists will be taken for training. 

When the flag *read_prefetched_Data* is set to False, for the above example, the script will read the csv file and select 25 random images from the top 50 artists along with their labels. These filenames and labels can be stored in a text file by setting the flag *saveData_to_file* to True. It will store the 1250 image filenames and corresponding labels. 

When the flag *read_prefetched_Data* is set to True, the script will fetch the 1250 image filenames and corresponding labels saved previously.	

During developing stage, when we want to compare a model's performance against different hyper parameters/architectures, it is useful to train/test on the same set of images. It is for this reason I have the option of writing the filenames and labels to text file so that the same images can be used since when reading from the csv file, I shuffle and select different images by the artists each time. 

The csv script can be used for the final training. 

The path to the csv or the text file can be set with the flag *data_file_path* in [train.py](train.py). 

Note: the text file is saved in the same directory as this script. The path is defined in the [getData.py](getData.py) script. 

The directory where the image files lie is set using the flag *DATA_DIR* in [train.py](train.py).

## Running instruction for the training routine:
This assumes TensorFlow is installed on the system. If you don't have GPU, please set the Flag *device_id* to */cpu:0* in [train.py](train.py).
If setting the flags in the script:
			
```bash
python train.py
```

If setting the flags through terminal/command promp (example on how to set the flags. Add/remove flags as needed):
		
```bash
 python train.py --num_of_artists 100 --num_of_images_per_artist 50 --read_prefetched_Data False
 ```
 
 ## Testing:
The [test.py](test.py) script runs the Siamese network on the test pairs.
		
The testing images are given in a csv file whose path is set with *data_file_path* in [test.py](test.py).	
The csv file provided contains 1048576 rows with 4 columns: index, image1, image2, sameArtist. For each of 1048576 test pairs, the csv contains the pair index( basically row count), filename for image 1 of the pair, filename for image 2 and their actual label(1 for same artist, 0 otherwise). 
		
The directory for the test images should be with the flag *DATA_DIR* in [test.py](test.py). The directory for the checkpints that are to be restored can be set with the flag *checkpt_dir* in [test.py](test.py).

## Running instruction for the test script:
This assumes TensorFlow is installed on the system. If you don't have GPU, please set the Flag *device_id* to */cpu:0* in [train.py](train.py).
If setting the flags in the script:
```bash
python test.py
```

If setting the flags through terminal/command prompt (example on how to set the flags. Add/remove flags as needed):
```bash
python test.py --data_file_path C:/Users/arbag/Documents/DeepLearning_in_CV/Project/Data/Data/submission_info.csv 
```

**Note:** Due to size constraints, I have not uploaded the dataset or the checkpoints.
