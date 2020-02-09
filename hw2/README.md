CS 1699: Homework 2
-------------------

**Due:** 2/11/2020, 11:59pm  
  
This assignment is worth 50 points.  
  
  
Part I: Computing weight updates by hand (10 points)  
  
In class, we saw how to compute activations in a neural network, and how to perform stochastic gradient descent to train it ([examples](neural_net_examples.pdf)). We computed activations for two example networks, but only showed how to train one of them. Show how to train the second network using just a single example, x = \[1 1 1\], y = \[0 0\] (note that in this case, the label is a vector). Initialize all weights to 0.05. Use a learning rate of 0.3. Include your answers in text form in a file report.pdf/docx.  
  
  
Part II: Training a neural network (20 points)  
  
In this exercise, you will write code to train and evaluate a very simple neural network. We will follow the example in Bishop Ch. 5 that uses a single hidden layer, a tanh function at the hidden layer and an identity function at the output layer, and a squared error loss. The network will have 30 hidden neurons (i.e. M=30) and 1 output neuron (i.e. K=1). To implement it, follow the equations in the slides and Bishop Ch. 5.  
  
First, write a function forward that takes inputs X, W1, W2 and outputs y\_pred, Z. This function computes activations from the front towards the back of the network, using fixed input features and weights. Also use the forward pass function to evaluate your network after training.  
  
**Inputs:**

*   an NxD matrix X of features, where N is the number of samples and D is the number of feature dimensions,
*   an MxD matrix W1 of weights between the first and second layer of the network, where M is the number of hidden neurons, and
*   an 1xM matrix W2 of weights between the second and third layer of the network, where there is a single neuron at the output layer

**Outputs:**

*   \[5 pts\] an Nx1 vector y\_pred containing the outputs at the last layer for all N samples, and
*   \[5 pts\] an NxM matrix Z containing the activations for all M hidden neurons of all N samples.

Second, write a function backward that takes inputs X, y, M, iters, eta and outputs W1, W2, error\_over\_time. This function performs training using backpropagation (and calls the activation computation function as it iterates). Construct the network in this function, i.e. create the weight matrices and initialize the weights to small random numbers, then iterate: pick a training sample, compute the error at the output, then backpropagate to the hidden layer, and update the weights with the resulting error.  
  
**Inputs:**

*   an NxD matrix X of features, where N is the number of samples and D is the number of feature dimensions,
*   an Nx1 vector y containing the ground-truth labels for the N samples,
*   a scalar M containing the number of hidden neurons to use,
*   a scalar iters defining how many iterations to run (one sample used in each), and
*   a scalar eta defining the learning rate to use.

**Outputs:**

*   \[5 pts\] W1 and W2, defined as above for forward, and
*   \[5 pts\] an itersx1 vector error\_over\_time that contains the error on the sample used in each iteration.

  
Part III: Testing your neural network on wine quality (20 points)  
  
You will use the [Wine Quality](http://archive.ics.uci.edu/ml/datasets/Wine+Quality) dataset. Use only the red wine data. The goal is to find the quality score of some wine based on its attributes. Write your code in a script neural\_net.py.

1.  \[10 pts\] First, download the winequality-red.csv file, load it, and divide the data into a training and test set using approximately 50% for training. Standardize the data, by computing the mean and standard deviation for each feature dimension using the train set only, then subtracting the mean and dividing by the stdev for each feature and each sample. Append a 1 for each feature vector, which will correspond to the bias that our model learns. Set the number of hidden units, the number of iterations to run, and the learning rate.
2.  \[3 pts\] Call the backward function to construct and train the network. Use 1000 iterations and 30 hidden neurons.
3.  \[3 pts\] Then call the forward function to make predictions and compute the root mean squared error between predicted and ground-truth labels, sqrt(mean((y\_test\_pred - y\_test).^2)). Report this number in a file report.pdf/docx
4.  \[4 pts\] Experiment with three different values of the learning rate. For each, plot the error over time (output by backward above). Include these plots in your report.

  
**Submission:** Please include the following files:

*   report.pdf/docx
*   neural\_net.py