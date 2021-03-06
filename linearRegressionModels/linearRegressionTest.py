import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
from linearRegression import linear_regression

test = linear_regression()
#original data plot
plt.scatter(test.population, test.profit) 
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
# Show the plot 
#plt.show() 

#variables for the cost function and gradient decent
m = len(test.population)
#convert data to np arrays, add column of ones to population data 
X = np.hstack((np.ones((m, 1)), np.array(test.population).reshape(m, 1)))
Y = np.array(test.profit).reshape(m, 1)
#initialize theta params to zero
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

#compute cost with cost function with theta values as zero
cost = test.compute_cost(X, Y, theta)
#print("first cost: ", cost[0])

#compute cost with new theta values
cost = test.compute_cost(X, Y, [-1, 2])
#print("second cost: ", cost[0])

#minimize the cost function using gradient decent, returns optimal theta values
[theta, costs] = test.gradient_decent(X, Y, theta, alpha, iterations)

#plot linear fit on top of training data
plt.plot(test.population, np.matmul(X, theta), 'r-')
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
        J_vals[i][j] = test.compute_cost(X, Y, t)

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


