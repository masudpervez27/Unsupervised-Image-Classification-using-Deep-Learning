# Unsupervised-Image-Classification-using-Deep-Learning

The objective of this work is to reveal the underlying patterns from image dataset. To achieve this goal I have applied an unsupervised Deep Learning method, specifically Convolutional Neural Networks (CNN) and analyze data. The reason it is unsupervised as we have unlabeled images which need to find some sort of patterns, like clustering. I have used a computer vision algorithm called “Convolutional Autoencoder” to solve this problem.

The architecture of this network design is symmetric about centroid and number of nodes reduce from left to centroid, they increase from centroid to right. Centroid layer would be compressed representation. 

The steps I have taken to complete this task describe shortly below:

## 1- Reshaping and rescaling
At first, I have loaded the image data and perform some preprocessing on it to fit for the network as our original input images have different shapes in size. In order to use the data for convolutional neural network, we need to get a fixed size format for all images. I choose an color input size of 224 px ×224px.

## 2- Train/ Test split
Second, split the dataset into train and test sets, where 70% of data (698 images) used for training and 30% of data (300 images) used for validation. 

## 3- Convolution Autoencoder
Next, I build a convolution autoencoder network where first layer is our input layer, followed by two CNN layers for reduction, next 2 layers are in charge of restoration. Final layer (output layer) restores images to same size of input. During the training process at each epoch, I set to print out relevant training and validation error by network. At the end of 50 epochs, I have found a satisfactory low loss by the model that provides a loss (mse) for training dataset is 0.0063 and validation dataset is 0.0061.

## 4- Clustering
Finally, I took the networks compressed representation layer (size: 56,56,8) , which takes a 6 times less space to original image (224, 224,3) and apply K-means clustering on it. For this I used our validation set because of limited computational resources.
The optimal number of clusters has chosen by using the elbow method on looking at the scree plot.  The plot suggests an optimal number to 6. The distributions of images into these 6 clusters are like this: 
70 images are in cluster 0, 
79 images are in cluster 1, 
41 images are in cluster 2, 
35 images are in cluster 3, 
16 images are in cluster 4 and finally 
59 images are in cluster 5. 

So, in this way both convolutional neural networks and autoencoder could be used to uncover underlying patterns from unlabeled image data. The main idea of autoencoder is to information reduction/dimensionality reduction from image data as like Principal Component Analysis, which is the pre-processing step for clustering. However unlike pca, an autoencoder can learn non-linear transformations.

## Places to Improve

The following are few ideas that could be possible to make improvement of the model performance as well as its pattern recognition.
-	Using Conv2D Transpose layers rather than Upscaling layers.
-	Try with different loss function such as “Perceptual Loss” instead of MSE.
-	Try other optimizer instead of “Adam” optimizer.
-	Using more Convolutional Layers with Dynamic Regularization.
-	Increasing training time, i.e., number of epochs.

P.S. The part of the analysis could be found in the provided corresponding Jupyter notebook.
