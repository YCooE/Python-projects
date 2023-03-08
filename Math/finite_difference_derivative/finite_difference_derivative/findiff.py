# Computing the finite difference derivatives of a 1D function represented by a N-dimensional vector of function values
# 1) input: f: a function represented as a N x 1 numpy array containing N equidistant function values sampled with time step deltat
#           deltat: a number describing the time step with which the x are sampled
#           h: A step size determining the level of approximation (h*deltat)
# 2) output: f_der: a N x 1 numpy array representing the centered finite difference approximation of f
#
# OBS! Remember to handle the boundaries by padding the original function with zeros

import numpy as np

def findiff(f, h, deltat):
    # recall: deltat is the x-axis interval covered by each increment h 
    #(the units for x, if you like)
    N = len(f)
    #Padd with zeros in both ends
    padds = np.zeros((1, h))[0]
    padded_f = np.concatenate((padds, f, padds), axis=0)
    
    #Centered finite diff
    f_der = (padded_f[2*h:2*h+N] - padded_f[0:N])/(2*h*deltat)
return f_de


## Case i) - f(x) = x^3 + 2x^2
# Function
def f(x_arr):
    f_vals = x_arr**3 + 2*x_arr**2
    return f_vals

# Analytical derivative
def f_der(x_arr):
    f_der_vals = 3*x_arr**2 + 4*x_arr
    return f_der_vals

xs = np.arange(0,10.1,0.1)
f_xs = f(xs)
f_der_num = findiff(f_xs, 1, 0.1) # Use findiff to compute centered finite 
der
f_der_ana = f_der(xs)
# Plot data, save and show plot


import matplotlib.pyplot as plt


plt.figure()
plt.plot(xs, f_xs)
plt.plot(xs, f_der_num)
plt.plot(xs, f_der_ana)
plt.axis([0, 10, 0, 500]) # Pick a sensible axis so that you can view the important part 
                          # of the derivative
plt.xlabel("x")
plt.ylabel("values")
plt.title('Plot of ' + r'$f(x) = x^3 + 2x^2$' + 'and its derivatives')
plt.legend(['$f(x)$', 'centered findiff', 'analytical der'], loc='upper left')
plt.savefig('case1_plot.png')
plt.show()


## Case ii) - growth data
# Load data and compute centered finite der
fdata = np.loadtxt('growthcurve.txt')
fdata_der_num = findiff(fdata, 1, 1)
#Here, function value increments are given one year apart, so deltat = 1
# Plot data, save and show plot
plt.figure()
plt.plot(fdata)
plt.plot(fdata_der_num)
plt.axis([0, 18, 0, 200])
plt.xlabel("x")
plt.ylabel("values")
plt.title('Plot of growth data with derivatives')
plt.savefig('case2_plot_both.png')
plt.show()