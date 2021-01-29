import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from logisticRegression import logisticRegression

test = logisticRegression()
#original data plot
fig = plt.figure(figsize=(10,7))
positive = [v[:-1] for x, v in enumerate(test.data) if test.data[x][2]]
negative = [v[:-1] for x, v in enumerate(test.data) if test.data[x][2] == 0]
plt.scatter([v[0] for x, v in enumerate(positive)], [v[1] for x, v in enumerate(positive)], c='r', marker='+') 
plt.scatter([v[0] for x, v in enumerate(negative)], [v[1] for x, v in enumerate(negative)], c='b', marker='_')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
# Show the plot 
#plt.show() 

#init variables
X = test.data[:, :-1]
Y = test.data[:, -1]
ones_X = np.hstack((np.ones((len(X), 1)), X))
initial_theta = np.zeros((len(X[0]) + 1, 1))
#compute cost and gradient using cost function.
cost, grad = test.costFunction(initial_theta, ones_X, Y)
#print(cost, grad)
#plot decision boundary
x_values = [np.min(X[:, 1] -3 ), np.max(X[:, 1] +3 )]
y_values = np.matmul((-1./grad[2]), (np.matmul(np.array(grad[1]).reshape((1, 1)),np.transpose(np.array(x_values).reshape((len(x_values), 1)))) + grad[0]))
plt.plot(x_values, -y_values[::-1], 'g-')
plt.show()