# Mathew Varughese
# MAV120

import numpy as np

# We will follow the example in Bishop Ch. 5 that uses 
# 	a single hidden layer, 
# 	tanh function at the hidden layer
# 	identity function at the output layer
# 	a squared error loss
# The network will have 
# 	30 hidden neurons (i.e. M=30)
# 	1 output neuron (i.e. K=1). 
# To implement it, follow the equations in the slides and Bishop Ch. 5.  

# This gets applied to the weights * the previous layer neurons
def activation_fn(weights_times_input):
	return np.tanh(weights_times_input)

# This gets applied to activations, which already had the
# the activation function applied to it
def compute_derivatives_from_activations(activations):
	# This is the derivative of tanh: 1 - (tanh x)^2
	return 1 - (activations ** 2)

# This function computes activations from the front towards the back of the network, using fixed input features and weights. 
# Also use the forward pass function to evaluate your network after training.

# Inputs:
# an NxD matrix X of features, where N is the number of samples and D is the number of feature dimensions,
# an MxD matrix W1 of weights between the first and second layer of the network, where M is the number of hidden neurons, and
# an 1xM matrix W2 of weights between the second and third layer of the network, where there is a single neuron at the output layer
# Outputs:
# [5 pts] an Nx1 vector y_pred containing the outputs at the last layer for all N samples, and
# [5 pts] an NxM matrix Z containing the activations for all M hidden neurons of all N samples.
def forward(X, W1, W2):
	"""
	X is N x D
	W1 is M x D
	Z is N x M

	Z contains the activations for the hidden layer. Each row is a new training example. 
	The "jth" column in Z corresponds to the "jth" activation in that network. 

	W2 is 1 x M
	y_pred is N x 1

	Every row in y_pred is a prediction for that training example.
	"""
	a = X @ W1.T
	Z = activation_fn(a)
	y_pred = Z @ W2.T 
	return (y_pred, Z)


# This function performs training using backpropagation (and calls the activation computation function as it iterates). 
# Construct the network in this function, i.e. create the weight matrices and initialize the weights to 
# small random numbers, then iterate: pick a training sample, compute the error at the output, then backpropagate 
# to the hidden layer, and update the weights with the resulting error.

# Inputs:
# an NxD matrix X of features, where N is the number of samples and D is the number of feature dimensions,
# an Nx1 vector y containing the ground-truth labels for the N samples,
# a scalar M containing the number of hidden neurons to use,
# a scalar iters defining how many iterations to run (one sample used in each), and
# a scalar eta defining the learning rate to use.
# Outputs:
# [5 pts] W1 and W2, defined as above for forward, and
# [5 pts] an itersx1 vector error_over_time that contains the error on the sample used in each iteration.
def backward(X, y, M, iters, eta):
	error_over_time = np.zeros((iters, 1))
	(N, D) = X.shape

	# Initialize weights to be random values
	W1 = np.random.normal(0.0, 1.0, (M, N))
	W2 = np.random.normal(0.0, 1.0, (M, 1))
	for i in range(iters):
		for n in range(N):
			(y_pred, Z) = forward(X, W1, W2)
			error_over_time[i] = 0.5 * (y_pred[n] - y[n]) ** 2
			# delta_k is matrix of backpropagation error for the output node. Since the activation
			# is the identity function, the formula for this is
			# delta_k = y_k - y_true
			delta_k = y_pred[n] - y[n]
			hidden_layer_activations = Z[n]
			# delta_j is a matrix of backpropagation error for the hidden layers. The activation 
			# function is tanh. d/dx tanh(x) = 1 - (tanh(x))^2
			# Z contains the activations of the functions after passed through tanh. So, we
			# know 1 - (tanh(x))^2 = 1 - Z^2, since Z = tanh(x).
			# So, we can called this derivative 'derived_h', since the Bishop book
			# refers to this value as h'(a).
			derived_h = compute_derivatives_from_activations(hidden_layer_activations)
			# delta_j = derived_h * sum_k w_kj * delta_k
			# Since there is only one output node, k=1, so this can be simplified to 
			# delta_j = derived_h * w_0j * delta_k
			# So, delta_j will be a 1 x M matrix that contains the error for each 
			# hidden neuron. The m'th item in this matrix will correspond to the
			# delta_j for that neuron. (Using delta_j since that is the formula in the 
			# Bishop book).
			delta_j = derived_h * W2 * delta_k
			# From the Bishop book, 
			# w_kj = w_kj - eta * delta_k * z_j
			# We calculate the change in this by multiply each activation 
			# by delta_k.
			W2_Delta = hidden_layer_activations * delta_k * eta;
			# w_ji = w_ji - eta * delta_j * x_i
			# X[n] is is of shape 1 x D
			# delta_j is of size 1 x M 
			# We want to create the weight deltas for the whole matrix
			# to be of size M x D. 
			# We broadcast the matrices to make element-wise multiplication
			# of X[n] and delta_j easier
			W1_Delta = np.broadcast_to(X[n], (M, D)) * np.broadcast_to(delta_j.T, (M, D))
			W1 -= W1_Delta
			W2 -= W2_Delta
	return (W1, W1, error_over_time)


def load_data(filename):
	np.genfromtxt(filename, delimiter=';')[1:]

load_data('winequality-red.csv')