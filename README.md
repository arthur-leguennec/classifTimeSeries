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
4. Run a script for testing differents models with differents hyperparameters.
5. Run this script on several dataset (like 15-20 for beginning) to obtain a scatter plot.
6. Understand how use the differents criterions
7. Save the seed while the initialisation for retrain the model in the same way.
8. Test several ways while the training and choose the one with the minimum current error.


## Models
The differents models are described in the file [model.lua](./model.lua).

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


### Observation
When we choose 0.1 for the learning rate, we have a problem when there is two classes.
