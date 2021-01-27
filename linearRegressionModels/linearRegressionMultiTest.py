import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from linearRegressionMulti import linearRegressionMulti

test = linearRegressionMulti()
X = test.data[:, :-1]
Y = test.data[:, -1]

#normalize data and generate mu and sigmas for data
x_norm, mu, sigma = test.normalizeFeatures(X)

#add column of ones to normalized features
x_norm = np.hstack((np.ones((len(X), 1)), x_norm))

#variables for gradient decent
alpha = 0.1
num_iterations = 400

#initialize theta and run grad decent to return best cost and theta parameters
theta = np.zeros((3, 1))
theta, cost_history = test.gradientDecentMulti(x_norm, Y, alpha, theta, num_iterations)
#print(theta)

#plot the cost function
plt.plot(cost_history, 'g-')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost using theta parameters')
plt.show()

#predict price of house with 1650 sq ft and 3 bedrooms
price1 = [theta[0, 0] * 1 + ((1650 - mu[0, 0])/sigma[0, 0]) * theta[1, 0] + ((3 - mu[0, 1])/sigma[0, 1]) * theta[2, 0]]
#print(price1)

#calculate theta values using normal equations without normalizing data
X = np.hstack((np.ones((len(X), 1)), X))
theta = test.normalEquation(X, Y)
#make prediction with theta values
price2 = theta[0] + theta[1] * 1650 + theta[2] * 3
#print(price2)
