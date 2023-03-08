# Numerically solve the growth-decay differential equation f'(t) = 0.15f(t), f(0) = 100
# 1) input: 	T:	float; the representing current time
#		k:	int; the number of bins
#
# 2) output: 	fT:	float; the number f(T) of bacteria at the current time T


import numpy as np

def diffeq(T, k):
# Solves the differential equation by 
# * first estimating the population at time T/k using your solution from a), and then 
# * iteratively estimating the population at time (i+1)*T/k from the population at time 
#   i*T/k until i+1=k, using your solution from b)
	init_f = 100
	F = np.zeros([k+1]) # Initialize the vector of function values to be filled in
	F[0] = init_f # Fill in the initial value
	Fder = np.zeros([k]) # Initialize the vector of derivatives to be filled in
	delta_t = T/k # Compute the timestep delta_t    
		
	for i in range(k):
		Fder[i] = cur_f_der(F[i])
		F[i+1] = approx_next_f(F[i], Fder[i], delta_t)
			
	fT = F[k]
	return fT

def cur_f_der(cur_f):
	# Computing the current derivative given the current function value
	return 0.15*cur_f


def approx_next_f(cur_f, cur_f_der, delta_t):
	# Computing an estimate of the next function value given the current function value 
	# (cur_f), the current derivative (cur_f_der) and the time step (delta_t)
	return cur_f + delta_t*cur_f_der


k = 100
T = 10
finalT = diffeq(T, k)
print(finalT)