import matplotlib.pyplot as plt
import numpy as np

# declare data
X = np.array([[i, i**2] for i in range(1,9)])
y = np.array([8.57, 14.33, 21.13, 24.29, 23.85, 25.19, 22.19, 15.96])
y = y.reshape((len(y),1))

# compute model (note: We do not augment the
# data matrix X (hence, no bias term w_0)
w = np.linalg.inv(np.dot(X.T,X))
w = np.dot(w, X.T)
w = np.dot(w, y)
print(w.shape)
print("Computed weights: %s" % str(w))

# calculating the horizontal distance of the landing point ...
landing_dist = -w[0]/ w[1]
print("Landing distance: %f" % landing_dist)

# plot model
x_test = np.array([[i,i**2] for i in np.linspace(0, 12, 100)])
result = np.dot(x_test, w) 
plt.plot(x_test[:,0], result, 'r--', label='Polynomial fit')
plt.plot(X[:, 0], y, 'bo', label='Datapoints')

# plot cannon
plt.annotate(
    'Cannon', xy=(0, -0.01), xytext=(-1.5, 5),
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'green', alpha = 0.5),
    arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0.3'))

# plot landing spot
plt.annotate(
    'Estimated landing spot', xy=(landing_dist,0), xytext=(landing_dist-1,4),
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.5),
    arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0.3'))

plt.title("Baron Munchhausen's canonball flight")
plt.xlabel("Horizontal distance")
plt.ylabel("Height")
plt.ylim( (0, 30) )
plt.xlim( (-2, 11) )
plt.legend()
plt.grid()
plt.show()
