###
###
import re
import matplotlib.pyplot as plt 
import numpy as np 

class linearRegression(object):
    
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

x = linearRegression()

#original data plot
plt.scatter(x.population, x.profit) 
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
# Show the plot 
#plt.show() 

#variables for the cost function and gradient decent
m = len(x.population)
#convert data to np arrays, add column of ones to population data 
X = np.hstack((np.ones((m, 1)), np.array(x.population).reshape(m, 1)))
Y = np.array(x.profit).reshape(m, 1)
#initialize theta params to zero
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

#compute cost with cost function with theta values as zero
cost = x.computeCost(X, Y, theta)
print("first cost: ", cost[0])

#compute cost with new theta values
cost = x.computeCost(X, Y, [-1, 2])
print("second cost: ", cost[0])

#minimize the cost function using gradient decent, returns optimal theta values
[theta, costs] = x.gradientDecent(X, Y, theta, alpha, iterations)

#plot linear fit on top of training data
plt.plot(x.population, np.matmul(X, theta), 'r-')
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
print("Prediction 1: ", predict1[0] * 10000)
print("Prediction 2: ", predict2[0] * 10000)

