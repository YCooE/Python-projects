# Numerically estimate an integral for a function specified at integer samples
# 1) input: 	sample_fun: 	A N x 1 numpy array representing N samples f(1), f(2), f(3), ... f(N)
#		x1: 		integer, the lower bound for the integral
#		x2:		integer, the upper bound for the integral
# 2) output: 	int_val:	float (a real number), the approximated value of the integral

import numpy as np

#Load the data
data = np.loadtxt("data/precipitation.txt")
N = data.size

def num_int(sample_fun, x1, x2):
# Numerically approximate the integral as a Riemann sum, as discussed in the lecture
   
# Makes sure the interval makes sense.
	assert (0 <= x1)
	assert (x1 <= x2)
	assert (x2 <= N)    
# Here we could check if x1 and x2 are integers.    
    
	int_val = 0
	# We want to include the x2'th.
	for t in range(x1, x2+1):
		int_val += sample_fun(t)
			
	# Comment in if you want to see the values.
	#print("day", t, "had", sample_fun(t), "mm percipation")
		
	return int_val



 # Precipitation functions
def p(t):
	return data[t]
def P(t):
	return num_int(p, 0, t)

P(182)

