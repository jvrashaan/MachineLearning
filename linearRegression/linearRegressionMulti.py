import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class linearRegressionMulti(object):
    #Given a data set of home features and home prices this class will predict the price of homes
    # given the features home size and number of bedrooms.
    def __init__(self):
        with open('ex1data2.txt') as f:
            data = f.read()
        
        self.data = np.array(re.split(',|\n', data)).reshape((47, 3)).astype(np.float)

    def normalizeFeatures(self, X):
        #returns a normalized version of X where
        #the mean value of each feature is 0 and the standard deviation
        #is 1.
        x_norm = X
        mu = np.zeros((1, len(X[0])))
        sigma = np.zeros((1, len(X[0])))
        features = len(X[0])
        for i in range(features):
            mu[0, i] = np.mean(x_norm, axis=0)[i]
            x_norm[:, i] = x_norm[:, i] - mu[0, i]
            sigma[0, i] = np.std(x_norm, axis=0)[i]
            x_norm[:, i] = x_norm[:, i] / sigma[0, i]
        
        return [x_norm, mu, sigma]

    def gradientDecentMulti(self, X, Y, alpha, theta, iterations):
        #Performs gradient descent to learn theta
        #updates theta by taking num_iters gradient steps with learning rate alpha
        #m = # of samples
        m = len(Y)
        J_history = np.zeros((iterations, 1))
        features = len(X[0])
        for i in range(iterations):
            temp = np.zeros((features, 1))
            for j in range(m):
                for k in range(features):
                    temp[k, 0] += (np.matmul(np.transpose(theta), np.transpose(X[j, :].reshape((1, features)))) - Y[j])[0, 0] *  X[j, k]
        
            for j in range(features):
                theta[j, 0] -= alpha * (1 / m) * temp[j, 0]

            # Save the cost J in every iteration    
            J_history[i] = self.computeCostMulti(X, Y, theta)
        
        return theta, J_history

    def computeCostMulti(self, X, Y, theta):
        #Compute cost for linear regression with multiple variables
        #m = # of samples
        m = len(Y)

        temp = 0
        for i in range(m):
            cost = np.matmul(np.transpose(theta), np.transpose(X[i, :]))[0]
            temp += (cost - Y[i]) ** 2
        
        return temp /(2 * m)

test = linearRegressionMulti()
X = test.data[:, :-1]
Y = test.data[:, -1]

#normalize data and generate mu and sigmas for data
x_norm, mu, sigma = test.normalizeFeatures(X)

#add column of ones to normalized features
x_norm = np.hstack((np.ones((47, 1)), x_norm))

#variables for gradient decent
alpha = 0.1
num_iterations = 400

#initialize theta and run grad decent to return best cost and theta parameters
theta = np.zeros((3, 1))
theta, cost_history = test.gradientDecentMulti(x_norm, Y, alpha, theta, num_iterations)
#print(theta, cost_history)

#predict price of house with 1650 sq ft and 3 bedrooms
price = [theta[0, 0] * 1 + ((1650 - mu[0, 0])/sigma[0, 0]) * theta[1, 0] + ((3 - mu[0, 1])/sigma[0, 1]) * theta[2, 0]]
print(price)
