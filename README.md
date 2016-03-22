# Classification Time Series

Project for my internship.
My goal is to test the different model of neural network for classify the data of UCR (see http://www.cs.ucr.edu/~eamonn/time_series_data/).
A lot of combination will be tested for find the parameters where the neural network works good.
The parameters (or hyperparameters in the case of convolutional) are the learning rate, the number of epochs, the size of the batch, the architecture of neural networks, the data type, ...

The project is separate in different files.

## Objectives :
1. Make a simple neural network. ![check](./images/check.png)
2. Make a convolutional neural network (like leNet). ![check](./images/check.png)
3. Make several neural network with differents parameters. ![check](./images/check.png)
4. Run a script for testing differents models with differents hyperparameters. (almost!, I have my script, but I don't have the power.)
5. Run this script on several dataset (like 15-20 for beginning) to obtain a scatter plot. (for now, I use LibreOffice)
6. Understand how use the differents criterions. ![check](./images/check.png)
7. Save the seed while the initialisation for retrain the model in the same way. (done! but I'm still working on it)
8. Test several ways while the training and choose the one with the minimum current error. (same above)
9. Run all tests on servers.
10. Save the training (error) to make a curve
11. Modify the class StochasticGradient for save the training, to have a learning rate variable, and lot of things!
12. Find a method (if it possible) for stop the training at good time.
 

## Models
The differents models are described in the file [model.lua](./model.lua).  
I defined just 4 or 5 models because my purpose isn't find the best convolutional neural network, but it's to test this CNN on time series and see if this neural networks work well or not.

## Training and Testing
The step of training and testing are separate in two files ([train.lua](./train.lua) and [test.lua](./test.lua)).

## Command line
There are differents command for run the file main.lua  
`-pathData`: command for the path with the data (train and test)  
`-fileTrain`: path for the train data (if there is a path in `pathData`, this command is ignored)  
`-fileTest`: path for the test data (if there is a path in `pathData`, this command is ignored)  
`-noModeCuda`: disable the mode cuda (enable by default)  
`-lr`: learning rate (0.01 by default)  
`-lrd`: learning rate decay (0 by default)  
`-iter`: max interation (15 by default)  
`-model`: choose the existing model (leNet1 by default)  
`-script`: if you want run a script for testing differents parameters, put this  

## Initialization
The way of initialization weights and biases is important for the training. In the paper "Understanding the difficulty of training deep feedforward neural networks" of Xavier Glorot and Yoshua Bengio, the authors say a good initialization for the weights is sqrt(6)/sqrt(n_(i) + n_(i+1)) with n the number of unit in layer, and initialize the biases at 0.

### Observation
When we choose 0.1 for the learning rate, we have a problem when there is two classes.  
I don't use the autoencoder for pretraining my neural network (with stacked autoencoder) because
