# **Traffic Sign Recognition** 

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

[image1]: ./examples/train_distribution.png "train distribution"
[image2]: ./examples/validation_distribution.png "validation distribution"
[image3]: ./examples/test_distribution.png "test distribution"
[image4]: ./examples/original.png "original image"
[image5]: ./examples/pca.png "pca augmented image"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

I have implemented LeNet, ResNet, and Inception net for this project, but I did not find a stable Inception net architecture for this project, therefore here I only list the project code link and test result with various data pre-processing and data augmentation combinations for LeNet and ResNet. Please be caustious that you need Tensorflow 1.3 to run them.

| NN model type | Data pre-processing | Data augmentation | Project Code	| Training accuracy | Validation accuracy | Test accuracy |
|:-------------:|:-------------------:|:-----------------:|:------------:|:-----------------:|:-------------------:|:-------------:| 
| LeNet | No | PCA color augmentation | [LeNet with PCA](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/51cc84db07623d89c549121c23b75f856edc2248/Traffic_Sign_Classifier.ipynb)	| 99.4% | 97.5% | 95.6% |
| LeNet | Input normalization | PCA color augmentation | [LeNet with PCA and Normalization](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/444ed68f5feba08c2fdc21bc0d1bef69bc209b7c/Traffic_Sign_Classifier.ipynb)	| 99.4% | 97.9% | 95.1% |
| ResNet | No | PCA color augmentation | [ResNet with PCA](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/46cceb25527c6c3f4762f00d85c6efe511c23d22/Traffic_Sign_Classifier.ipynb)	| 99.8% | 98.5% | 97.6% |
| ResNet | Input Normalization | PCA color augmentation | [ResNet with PCA and Normalization](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/afac1af269188e5fa60c6d8ab50477469f663851/Traffic_Sign_Classifier.ipynb)	| 99.8% | 98.6% | 98.0% |

BTW, I also implemented a [LeNet in Tensorflow 0.12](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). It just passes the validation accuracy requirement of the project rubrics, and I did not pay as much attention on it as on the other ones listed in the table above. This project report is also a summary of the ones in the table instead of the basic LeNet implememented in Tensorflow 0.12.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First, I showed a sample picture of each label class from the training set. You can check details in any project code in the table on the top. By doing so, I got an rough idea of what these images look like.
Second, I gathered the distribution of label class on training set, validation set, and test set, which almost share a same distribution.

![alt text][image1]
![alt text][image2]
![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For data pre-processing, I only tried one technique **input normalization**. The udacity lecture, this [StackExchange question](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c) and [CS231n of Stanford](http://cs231n.github.io/neural-networks-2/#datapre) all mention that **input normalization** is a necessary step of image recognition task, which helps control the gradients as well as the feature value in an effective range, as they all share the same weights and biases of the neural network. I do the input normalization in the common way that for all images in training, validation, and test sets, I use the mean and standard deviation of training set images to subract and divide. 

For data augmentation, I tried the **PCA color augmentation** idea mentioned in [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). According to my understanding, I computed the eigenvectors and eigenvalues of the 3x3 RGB channel covariance matrix for each image in training set, and then "add multiples of the found principal components, with magnitudes proportional to the corresponding eignevalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1". The PCA color augmentation strengthens or weakens the color feature of the image, while it keeps the patten and distribution of the training image. In addition, I implemented a **multithread data loading pipeline** which does the PCA color augmentation for the next mini-batches of training data by a CPU thread, and in the meantime let GPU thread execute the training. Namely, no additional compuation cost on GPU is added, so the training time is not increased due to data augmentation. The pipeline is like this: training data loading(CPU) -> PCA color augmentation(CPU) -> Enqueue(CPU) -> Dequeue(CPU) -> Train model(GPU)

**Original image**

![alt text][image4]

**PCA color augmented image**

![alt text][image5]

I did not try the affine transformation for data pre-processing or data augmentation for following reasons:
* For comparing different models, I spent most of my time on reading papers, implementing models, and other supporting techniques, like the multithread data loading pipeline and the tensorboard for tensorflow visualization and debugging, so I do not have enough time to try out these image processing techniques which I have already tried in the lane line finding project.
* Some of the affine transformations are not suitable for traffic sign recognition task, e.g. flip(mirror) could not be used in this case, because most mirror of the traffic signs are not traffic signs any more.
* Sophisticated data pre-processing should be avoided as much as possible, so that the model could be deployed to production more easily. Namely when the model is used to classify real traffic sign, it does not need to do the same data pre-processing steps before input into model, which could save cost of the classifier. According to my test result, [ResNet with PCA](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/46cceb25527c6c3f4762f00d85c6efe511c23d22/Traffic_Sign_Classifier.ipynb) already achieves a test accuracy of 97.6%, which is a qualified example without any data pre-processing steps required.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final **LeNet model** consists of the following layers:

| Layer         		         |     Description	        					                 | 
|:------------------------:|:---------------------------------------------:| 
| Input         		         | 32x32x3 RGB image   							                   | 
| Convolution 5x5     	    | 1x1 stride, valid padding, outputs 28x28x6  	 |
| ReLU                     |                                               |
| Batch Normalization      |                                               |
| Max pooling 2x2          | 2x2 stride, outputs 14x14x6                   |
| Convolution 5x5     	    | 1x1 stride, valid padding, outputs 10x10x16 	 |
| ReLU                     |                                               |
| Batch Normalization      |                                               |
| Max pooling 2x2          | 2x2 stride, outputs 5x5x16                    |
| Flatten                  | outputs 400                                   |
| Fully Connected          | outputs logits of 43 classes                  |
| Softmax                  | outputs softmax probablity of 43 classes      |


My final [ResNet model](https://arxiv.org/pdf/1512.03385.pdf) consists of the following layers:

| Layer         		         |     Description	        					                 | 
|:------------------------:|:---------------------------------------------:| 
| Input         		         | 32x32x3 RGB image   							                   | 
| Convolution 3x3     	    | 1x1 stride, same padding, outputs 32x32x16 	  |
| Residual module x 2      | 3x3 stride, outputs 32x32x16                  |
| Residual module x 2      | 3x3 stride, outputs 16x16x32                  |
| Residual module x 2      | 3x3 stride, outputs 8x8x64                    |
| Global average pooling   | outputs 1x1x64                                |
| Fully Connected          | outputs logits of 43 classes                  |
| Softmax                  | outputs softmax probablity of 43 classes      |

Roughly, **Residual module** consists of the following layers:

| Layer         		         |     Description	        					                 | 
|:------------------------:|:---------------------------------------------:| 
| Input     	              | The "x" in the last step 	                    |
| Convolution 3x3     	    | same padding                              	   |
| Batch Normalization      |                                               |
| ReLU                     |                                               |
| Dropout                  | Dropout rate 0.5                              |
| Convolution 3x3          | same padding, 1x1 strides,                    |
| Batch Normalization      | Output F(x)                                   |
| H(x) = F(x) + x          | outputs softmax probablity of 43 classes      |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Hyperparameter         		| Final Value      					                        | 
|:------------------------:|:---------------------------------------------:| 
| Initializer              | xavier_initializer                            |
| Optimizer     	          | Adam                                          |
| Learning rate       	    | 1e-3                                      	   |
| Regularization scale     | 1e-3 ~ 5e-2                                   |
| Dropout rate             | 0.5 (added by me, not mentioned in the paper) |
| Batch size               | 64                                            |
| Learning rate decay rate | 0.95 (decay after every epoch)                |
| Epochs                   | 100                                           |
| Queue capacity           | 512 (used in multithread data pipeline)       |

I put the train module in a nested loop of potential hyperparameters and execute fine tuning.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My best final model results were:
* training set accuracy of 99.8% 
* validation set accuracy of 98.6%
* test set accuracy of 98.0% 

| NN model type | Data pre-processing | Data augmentation | Project Code	| Training accuracy | Validation accuracy | Test accuracy |
|:-------------:|:-------------------:|:-----------------:|:------------:|:-----------------:|:-------------------:|:-------------:| 
| LeNet | No | PCA color augmentation | [LeNet with PCA](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/51cc84db07623d89c549121c23b75f856edc2248/Traffic_Sign_Classifier.ipynb)	| 99.4% | 97.5% | 95.6% |
| LeNet | Input normalization | PCA color augmentation | [LeNet with PCA and Normalization](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/444ed68f5feba08c2fdc21bc0d1bef69bc209b7c/Traffic_Sign_Classifier.ipynb)	| 99.4% | 97.9% | 95.1% |
| ResNet | No | PCA color augmentation | [ResNet with PCA](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/46cceb25527c6c3f4762f00d85c6efe511c23d22/Traffic_Sign_Classifier.ipynb)	| 99.8% | 98.5% | 97.6% |
| ResNet | Input Normalization | PCA color augmentation | [ResNet with PCA and Normalization](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/afac1af269188e5fa60c6d8ab50477469f663851/Traffic_Sign_Classifier.ipynb)	| 99.8% | 98.6% | 98.0% |


**LeNet summary:**
The LeNet model suggested in Udacity lecture is actually working pretty well. It trained pretty fast. On AWS GPU instance, it usually takes 4 seconds to go through one epoch of training data. While the cons of the model is that it is not very good at resist overfitting. If I choose a moderate dropout rate, e.g. 0.5, with a moderate L2 regularization scale, e.g. 3e-2, it would finally converges to the result you see in the table above. And if I choose a more aggressive dropout rate, e.g. 0.7, then the training rate will drop to 97%, and the validation and test rate will not exceed it as well. And also normalization and PCA could not really help a lot on mitigating the overfitting issue. The other problem of this model is that it is not generalized so well (which is also a symptom of overfitting), in the 5 new image test section, I usually got 80% accuracy, althought the test accuracy is approximately 95%.

**ResNet summary:**
So I tried the ResNet, because it is the most famous one right now. At first I tried the shallow network version mentioned in the [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf). Then I got even more overfitting than LeNet, so I tried to add a dropout layer with dropout rate 0.5 in each Residual module. Then the test result became pretty good. The training time is problem of this ResNet model, whose training time per epoch is about 28 seconds, almost 7 times of LeNet's. But in the end, it raises test accuracy for about 3%, and it always get 100% on the 5 new images test.

**Inception net summary:**
I also tried to implement an Inception net, but did not find a proper one for this task, either overfitting or underfitting. If you have interest, you can have a look at my raw [Inception net model](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/Inception-tf-1.3/Traffic_Sign_Classifier.ipynb).


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


