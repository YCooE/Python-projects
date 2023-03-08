# Use gradient descent to minimize the mean Huber error for a given dataset.
# 1) input: dataset: A N x 1 numpy array representing a dataset of numbers read in as dataset = np.loadtxt('outlier.txt')
# 2) output: der: a 1 x 1 numpy array representing the obtained Huber mean of the dataset

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt



def gradient_descent_huber(dataset):
	# Implement gradient descent for minimizing the mean Huber error for the dataset. 
	# Tip: Separate out the functions for the datapoint-wise and dataset-wise Huber 
	# errors as indicated below.
	
	# Initialization
	cur_alpha = np.mean(dataset) # initialize at the mean (could be random, 0, whatever) 
	convergence = False
	max_iter = 10000
	learningrate = 0.01
	conv_thr = 0.0001
	num_iter = 0
	while not(convergence):
	# compute the derivative at the current alpha
	alpha_der = huber_error_dataset_der(dataset, cur_alpha)
	#print([cur_alpha, alpha_der])
	
	# set step length 
	step = -learningrate*alpha_der
	
	# take a step in the stepping direction
	cur_alpha = cur_alpha + step
	
	# check for convergence/stopping
	if num_iter == max_iter:
		print('reached max_iter')
		convergence = True
	elif np.linalg.norm(alpha_der) < conv_thr:
		convergence = True
	
	# update counter
	num_iter+=1
	
	optimal_alpha = cur_alpha
	#When we are reach "convergence", our current alpha is returned as the optimal one
	return optimal_alpha


def huber_error_dataset_der(dataset, alpha):
	# Compute the derivative of the mean Huber error function; note that this is the mean of the data point-wise Huber error derivatives
 
	N = len(dataset)
	
	# Initialize np array for datapoint-wise errors
	huber_der = np.zeros(N)
	for i in range(N):
		huber_der[i] = huber_error_der(dataset[i], alpha)
	total_error_der = np.sum(huber_der)
	return total_error_der


def huber_error_der(x, alpha):
	# Compute the datapoint-wise Huber error derivative for the single datapoint x at the value alpha
	der = 2 *(alpha - x)

	return der



