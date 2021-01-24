import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import linearRegression as linearReg

class linearRegressionMulti(object):
    #Given a data set of home features and home prices this class will predict the price of homes
    # given the features home size and number of bedrooms.
    def __init__(self):
        with open('ex1data2.txt') as f:
            data = f.read()
        
        self.data = np.array(re.split(',|\n', data)).reshape((47, 3))

    def normalizeFeatures(self, X):
        #returns a normalized version of X where
        #the mean value of each feature is 0 and the standard deviation
        #is 1.
        x_norm = X
        mu = np.zeros((1, len(X[0])))
        sigma = np.zeros((1, len(X[0])))
        features = len(X[0])
        for i in range(features):
            mu[1][i] = np.mean(x_norm[:][i])
            x_norm[:][i] = x_norm[:][i] - mu[1][i]
            sigma[1][i] = np.std(x_norm[:][i])
            x_norm[:][i] = x_norm[:][i] /  sigma[1][i]
        
        return [x_norm, mu, sigma]
        
    def runLinearRegressionMulti(self):
        X = self.data[:][:-1]
        Y = self.data[:][-1]
        print(X, Y)

lrm = linearRegressionMulti()
X = lrm.data[:][:-1]
Y = lrm.data[:][-1]

#normalize data
x_norm, mu, sigma = lrm.normalizeFeatures(X)
print(x_norm, mu, sigma)
