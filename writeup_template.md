# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./project_report/histogram.png "Visualization"
[image2]: ./project_report/before_image.png "Visualization"
[image3]: ./project_report/augmented_image.jpg "Visualization"
[image4]: ./real_images/Annotation1.jpg "Traffic Sign 1"
[image5]: ./real_images/Annotation2.jpg "Traffic Sign 2"
[image6]: ./real_images/Annotation3.jpg "Traffic Sign 3"
[image7]: ./real_images/Annotation4.jpg "Traffic Sign 4"
[image8]: ./real_images/Annotation5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

```python
# Load pickled data
import pickle

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = train['features'].shape[0]
n_validation = valid['features'].shape[0]
n_test = test['features'].shape[0]
image_shape = train['features'][0].shape
n_classes = y_train.max() + 1
```

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I actually did not perform any preprocessing to the images; however, I did implement some image augmentation. This included 

| Original        | Augmented   | 
|:-------------:|:-------------:| 
| ![alt text][image2] | ![alt text][image3] | 

The difference between the original data set and the augmented data set is the following, the image has:
* a rotation range of [-15, 15] degrees, 
* a zoom of [1.15, 0.85] image zoom, 
* a horizontal shift of [-0.1,0.1],
* a vertical shift of [-0.1,0.1], 
* a shear (a tilt in image) [1.15, 0.85]
* a brightness level change from [.2, 1] 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer (type)                 | Output Shape             | Param #   |
|:----------------------------:|:------------------------:|:---------:| 
| conv2d (Conv2D)              | (None, 32, 32, 8)        | 224       |      
| batch_normalization (BatchNo | (None, 32, 32, 8)        | 32        |
| max_pooling2d (MaxPooling2D) | (None, 16, 16, 8)        | 0         | 
| conv2d_1 (Conv2D)            | (None, 16, 16, 16)       | 1168      |  
| batch_normalization_1 (Batch | (None, 16, 16, 16)       | 64        |  
| max_pooling2d_1 (MaxPooling2 | (None, 8, 8, 16)         | 0         |    
| conv2d_2 (Conv2D)            | (None, 8, 8, 32)         | 4640      |   
| batch_normalization_2 (Batch | (None, 8, 8, 32)         | 128       |   
| max_pooling2d_2 (MaxPooling2 | (None, 4, 4, 32)         | 0         | 
| flatten (Flatten)            | (None, 512)              | 0         | 
| dense (Dense)                | (None, 256)              | 131328    |
| batch_normalization_3 (Batch | (None, 256)              | 1024      |  
| dropout (Dropout)            | (None, 256)              | 0         |
| dense_1 (Dense)              | (None, 128)              | 32896     | 
| batch_normalization_4 (Batch | (None, 128)              | 512       | 
| dropout_1 (Dropout)          | (None, 128)              | 0         | 
| dense_2 (Dense)              | (None, 43)               | 5547      |  
_________________________________________________________________
Total params: 177,563

Trainable params: 176,683

Non-trainable params: 880
_________________________________________________________________
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an:
* An Adam Optimizer
  * A learning rate of 0.01
  * A decay of 0.0003
    * This is used to decrease the learning rate as the algorithm learns
* A batch size of 64
* An epoch of 100
  * I added an early termination function so that once the validation accuracy starts plateauing it terminates the training
* For the model itself
  * A dropout rate of 0.5
  * A L2 rate of 0.0001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of `0.9390`
* validation set accuracy of `0.9290`
* test set accuracy of `0.9690`

My approach to finding the architecture:

* I started researching online for some model architecture to start with. I found [this PySearch blog](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/#:~:text=Traffic%20sign%20classification%20is%20the,to%20build%20%E2%80%9Csmarter%20cars%E2%80%9D.) and started to try using the architecture introduced.
* I then did some changes and added an extra convolution layer as well as the L2 regularization to remove the overfitting
* I then added learning rate decay in order to improve the accuracy
* I then tried interesting though batch sizes in order to see what worked best 64 seemed to work best

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, I used google maps and went though the streets to find traffic signs in Germany:

| ![alt text][image4] | ![alt text][image5] | ![alt text][image6] |
|:-------------------:|:-------------------:|:-------------------:| 
| ![alt text][image7] | ![alt text][image8] | |

The quality of the images are good; however they needed to be rescaled in order to be able to pass them into the network. Hardest image to classify would be the 60km/h because it so so rotated. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	   	| 
|:---------------------:|:---------------------:| 
| No entry      		| No entry   			| 
| Children crossing     | Children crossing		|
| General caution		| General caution		|
| Speed limit (50 km/h)	| Speed limit (50 km/h)	|
| Speed limit (60 km/h)	| Speed limit (30 km/h)	|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. To improve the model I would need to create more randomly rotated images so that it is easier to detect. I would also need more data in order to increase the dataset and make it better model.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a no entry (probability of 1.0), and the image is a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 1.0         			| No entry   									    | 
| 0.0     				| Stop 										        |
| 0.0					| Priority road										|
| 0.0	      			| Yield					 				            |
| 0.0				    | End of no passing by vehicle over 3.5 metric tons	|


For the second image, the model is very sure that this is a children crossing (probability of 1.0), and the image is a children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        	| 
|:---------------------:|:-----------------------------:| 
| 1.0         			| Children crossing   			| 
| 0.0     				| Bicycles crossing 			|
| 0.0					| Beware of ice/snow			|
| 0.0	      			| Slippery road					|
| 0.0				    | Dangerous curve to the right	|


For the third image, the model is pretty sure that this is a general caution (probability of 0.935), and the image is a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        	| 
|:---------------------:|:-----------------------------:| 
| 0.935        			| General caution   			| 
| 0.026    				| Pedestrians 			        |
| 0.013					| Dangerous curve to the right	|
| 0.01	      			| Traffic signals				|
| 0.008				    | Road work                 	|


For the fourth image, the model is pretty sure that this is a speed limit of 50km/h (probability of 0.449), and the image is a speed limit of 50km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        	| 
|:---------------------:|:-----------------------------:| 
| 0.449        			| Speed limit of (50km/h)  		| 
| 0.261    				| Dangerous curve to the left 	|
| 0.088					| Speed limit of (30km/h)	    |
| 0.074	      			| Double curve				    |
| 0.036				    | Slippery road                 |

For the fourth image, the model is pretty sure that this is a speed limit of 30km/h (probability of 0.959), and the image is a speed limit of 60km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        	| 
|:---------------------:|:-----------------------------:| 
| 0.959        			| Speed limit of (30km/h)  		| 
| 0.034    				| Speed limit of (50km/h)    	|
| 0.005					| Speed limit of (20km/h)	    |
| 0.001	      			| Speed limit of (80km/h)       |
| 0.0				    | Speed limit of (70km/h)       |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


