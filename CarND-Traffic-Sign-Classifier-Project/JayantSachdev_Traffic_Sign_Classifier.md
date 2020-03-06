# **Traffic Sign Recognition** 
**Jayant Sachdev**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/LoadImages.png "LoadImages"
[image2]: ./results/TrainingDist.png "Distribution of Training Dataset"
[image3]: ./results/ValidDist.png "Distribution of Validation Dataset"
[image4]: ./results/TestingDist.png "Distribution of Testing Dataset"
[image5]: ./results/Training_ValidationHistory.png "Graph of Training and Validation Accuracy through each EPOCH of Training"
[image6]: ./results/German_Traffic_Signs.png "German Traffic Signs"
[image7]: ./results/PerformanceofNN.png "Performance of the Final Trained Network"


In this Project, i utilized deep learning and convolutional neural networks to classify traffic signs from Germany. My implementation and project code can be reviewed [here](https://github.com/JSachdev92/TrafficSignClassifier/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

There are 43 unique road signs that we want to classify in this project. In order to visualize and understand what kind of signs were chosen, i plotted the first image of each road sign. 

![alt text][image1]

I then decided to plot a bar graph showing how many images we have of each road sign in the training dataset to obtain a distrubition:
![alt text][image2]
I then did the same for the validation and testing datasets and noticed that the general distribution appeared similar. I believe this is important as a mismatched distribution will mean that the training dataset should be modified to reflect the distributions of our validation and testing datasets for best results:
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### Data Pre-Processing 


For the dataprocessing, i just normalized the pixel data between 0 and 1. This is one area i would like to spend more time on when i revisit the project as i feel that adding data and using pre-processing techiniques like converting to greyscale will yield even better results. 


#### Designing the Model Architecture and design decisions

In terms of the model architecture, i started with the LeNet architecture from the CNN lesson and adjusted the input information and final output information to get it to run. I started with the LeNET architecture based on the directions given in the project and in the course materials. The LeNET architecture utilizes Convolutional Neural Networks (CNN's) in sequence, connected to fully connected layers to process information. I think it is good for a traffic sign classification problem because CNN's are known to be useful to detect information in an image since it breaks down the image into smaller images, moves over the whole image and creates a new image with the results. This allows it to learn individual defining aspects of images, without human supervision and with minimal computational cost. Initially I took the LeNET architecture and tried to play with the hyperparameters, specifically the learning rate, EPOCH and batch size. I managed to obtain a 92% validation accuracy which was higher than i was expecting. I then realized that there was some definite overfitting occuring, so i implemented a basic L2 regularization, adding a cost for the weights at every step of the network. After tuning the hyper parameters, i achieved a 93% validation accuracy with this approach. I then decided to replace the L2 regularization with the dropout technique taught in the classroom. I initially put this on all the layers except the final output and achieved a validation accuracy of 95%. 
I then reduced the number of dropouts until i found an optimal solution with 1 dropout in the 2nd convolution layer and  1 dropout in the 1st fully connected layer. This allowed me to improve my validation accuracy to 96%.
I then added another CNN layer, adjusted the dropout probability and utilized and adaptive learning rate so that i can use a high rate at the begining and a lower learning rate towards the end of the training cycle and achieved a final learning rate of: 96.6%

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 14x14x6|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16|
| RELU					|												|
| Dropout		|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 |
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 2x2x200|
| RELU					|												|
| Flatten					|	800 outputs	|
| Fully connected		| 120 Outputs	|
| Dropout		|												|
| Fully connected		| 84 Outputs	|
| Fully connected		| 43 Outputs	|
| Softmax				|         									|

 


#### The training process

Once i had finalized the architecture above, the training process was fairly straight forward. I played with several parameters, including batch size, epoch, learning rate and the keep probability for the dropout functions. For the Batch size, i played with the batch size, starting from 8 and going all the way up to 500. I found that as i reduced the batch size, the predicitions were slightly more accurate at the cost of speed. I came upon a good comprimise at a batch size of 200. In terms of the dropout probability, i started with a keep probability of 50% and played with the values, going up to 75% and going all the way down to 30%. When i raised the probability, i noticed more underfitting on the data. when i raised the probability, i noticed the network had a harder time learning the images. I noticed that the optimal performance was with a keep rate of 50% and 57.5% and i settled on a keep rate of 55%. For the learning rate, i noticed that a smaller learning rate resulted in the model predicition having less of a variance between each iteration. This meant that the parameters were adjusted by smaller amounts each time and the results would be more stable. The downside was that it required more epochs to reach the same overall prediction accuracy as a higher learning rate. Having a higher learning rate reached the ballpark predicition accuracy faster, but then resulted in overfitting at higher EPOCH's. To obtain the best of both worlds, i implemented a variable learning rate that slowed down with time. I started with a learning rate of 0.003 and as it approached EPOCH 100, it would be nearing a learning rate 5 times less than the original. In terms of the EPOCH's, i picked a value of 150 as i realized that there were fairly noticable improvements made in the 125-145 EPOCH range and wanted the system to stabalize to a solution. I could have probably chosen 200 EPOCH's to allow for an even more stable solution but i found this worked well.

The training and validation accuracy as i trained the model are shown in the figure below:
![alt text][image5]
 
My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.966 
* test set accuracy of 0.949


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


