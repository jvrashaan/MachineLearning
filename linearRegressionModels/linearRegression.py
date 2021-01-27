import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class linearRegression():
    #Given a data set containing the population of cities and the profit made in those cities for a food truck
    # this class will predict how much profit can be made in other cities with given populations
    def __init__(self):
        with open('ex1data1.txt') as f:
            data = f.read()
        
        data = re.split(',|\n', data)
        
        self.population = [float(data[x]) for x in range(0, len(data), 2)]
        self.profit = [float(data[y]) for y in range(1, len(data), 2)]

    def computeCost(self, X, Y, theta):
        #computes the cost of using theta as the
        #parameter for linear regression to fit the data points in X and y
        m = len(Y)
        cost = 0

        temp = 0
        for sample in range(m):
            temp += ((theta[0] + theta[1] * X[sample, 1]) - Y[sample]) ** 2
        
        return temp/ (2 * m)

    def gradientDecent(self, X, Y, theta, alpha, iterations):
        #minimizes the cost function given parameters theta.
        m = len(Y)
        costs = np.zeros((iterations, 1))
        for i in range(iterations):
            #inner loop that implements gradient decent per interation
            temp1 = 0
            temp2 = 0
            for x in range(m):
                temp1 += ((theta[0] + theta[1] * X[x, 1]) - Y[x]) * X[x, 0]
                temp2 += ((theta[0] + theta[1] * X[x, 1]) - Y[x]) * X[x, 1]
        
            theta[0] = theta[0] - alpha * (1 / m) * temp1
            theta[1] = theta[1] - alpha * (1 / m) * temp2
            #save cost on every iteration
            costs[i] = self.computeCost(X, Y, theta)
        
        return [theta, costs]
