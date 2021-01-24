import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class linearRegression(object):
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

    def runLinearRegression(self):
        #original data plot
        plt.scatter(self.population, self.profit) 
        plt.xlabel('Population of City in 10,000s')
        plt.ylabel('Profit in $10,000s')
        # Show the plot 
        #plt.show() 

        #variables for the cost function and gradient decent
        m = len(self.population)
        #convert data to np arrays, add column of ones to population data 
        X = np.hstack((np.ones((m, 1)), np.array(self.population).reshape(m, 1)))
        Y = np.array(self.profit).reshape(m, 1)
        #initialize theta params to zero
        theta = np.zeros((2, 1))
        iterations = 1500
        alpha = 0.01

        #compute cost with cost function with theta values as zero
        cost = self.computeCost(X, Y, theta)
        #print("first cost: ", cost[0])

        #compute cost with new theta values
        cost = self.computeCost(X, Y, [-1, 2])
        #print("second cost: ", cost[0])

        #minimize the cost function using gradient decent, returns optimal theta values
        [theta, costs] = self.gradientDecent(X, Y, theta, alpha, iterations)

        #plot linear fit on top of training data
        plt.plot(self.population, np.matmul(X, theta), 'r-')
        plt.legend(["Linear regression", "Training data"])
        plt.show()

        #plot the cost function
        plt.plot(costs, 'g-')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost using theta parameters')
        plt.show()

        #predict profit in cities with 35K and 70K population
        predict1 = np.matmul([1, 3.5] ,theta)
        predict2 = np.matmul([1, 7.0], theta)
        #print("Prediction 1: ", predict1[0] * 10000)
        #print("Prediction 2: ", predict2[0] * 10000)

        #plotting cost function given theta values on surface and contour plots
        theta0_vals = np.linspace(-10, 10, 100)
        theta1_vals = np.linspace(-1, 4, 100)

        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

        for i in range(len(theta0_vals)):
            for j in range(len(theta1_vals)):
                t = [theta0_vals[i], theta1_vals[j]]
                J_vals[i][j] = self.computeCost(X, Y, t)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True)
        plt.xlabel('Theta 0')
        plt.ylabel('Theta 1')
        plt.show()

        #contour plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.contour(theta0_vals, theta0_vals, J_vals)
        plt.xlabel('Theta 0')
        plt.ylabel('Theta 1')
        plt.show()


