import numpy
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
mean = t_train.mean()
print("Mean price: %f" % mean)

# (b) RMSE function
def rmse(t, tp):
    return numpy.sqrt(numpy.mean((t - tp)**2))
# mean predictions
preds = mean * numpy.ones(len(t_test))
err = rmse(preds, t_test)
print("RMSE using mean predictor: %f" % err)

# (c) visualization of results
plt.scatter(t_test, preds)
plt.xlabel("House Prices")
plt.ylabel("Predicted House Prices")
plt.xlim([0,50])
plt.ylim([0,50])
plt.title("Mean Estimator (RMSE=%f)" % err)
plt.show()
