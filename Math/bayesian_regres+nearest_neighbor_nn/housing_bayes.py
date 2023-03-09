import numpy
import linreg
import matplotlib.pyplot as plt

def rmse(t, tp):
    """ Computes the RMSE for two
    input arrays 't' and 'tp'.
    """

    # sanity check: both shapes have to be the same!
    assert tp.shape == tp.shape

    return numpy.sqrt(numpy.mean((t - tp)**2))

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
t_train = t_train.reshape(len(t_train),1)
t_test = t_test.reshape(len(t_test),1)
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (1) fit regularised linear regression model
model_reglinreg = linreg.LinearRegression(lam=0.1)
model_reglinreg.fit(X_train, t_train)
preds_reglinreg = model_reglinreg.predict(X_test)
rmse_reglinreg = rmse(t_test, preds_reglinreg)

# (2) fit Bayesian linear regression model
model_bayeslinreg = linreg.BayesianRegression()
model_bayeslinreg.fit(X_train, t_train)
preds_bayeslinreg = model_bayeslinreg.predict(X_test)
rmse_bayeslinreg = rmse(t_test, preds_bayeslinreg)

# visualize results
plt.scatter(t_test, preds_reglinreg, label="Regularised Linear Regression RMSE: %f" % rmse_reglinreg)
plt.scatter(t_test, preds_bayeslinreg, label="Bayesian Linear Regression RMSE: %f" % rmse_bayeslinreg)
plt.xlabel("House Prices")
plt.ylabel("Predicted House Prices")
plt.xlim([0,50])
plt.ylim([0,50])
plt.title("Bayesian vs. Regularised Linear Regression")
plt.legend()
plt.show()
