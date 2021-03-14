import matplotlib.pyplot as plt 
import numpy as np 
from logisticRegression import logistic_regression

test = logistic_regression()
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
cost, grad = test.cost_function(initial_theta, ones_X, Y)
#print("cost: ", cost)
#print("grad:" , grad)
#plot decision boundary
x_values = [np.min(X[:, 1] -10 ), np.max(X[:, 1] +10 )]
y_values = np.matmul((-1./grad[2]), (np.matmul(np.array(grad[1]).reshape((1, 1)),np.transpose(np.array(x_values).reshape((len(x_values), 1)))) + grad[0]))
plt.plot(x_values, -y_values[::-1], 'g-')
plt.show()

#predict admission for student with following test scores
theta = np.array([-25.1613, 0.2062, 0.2015]).reshape((3, 1))
prob = test.sigmoid(np.matmul(np.array([1,45, 85]).reshape((1, 3)), theta))
#print("probability of sample student: ", prob[0][0])

#compute accuracy on entire training set
p = test.predict(theta, ones_X) 
#print("model accuracy: ", p)

#second plot
fig = plt.figure(figsize=(10,7))
positive = [v[:-1] for x, v in enumerate(test.data2) if test.data2[x][2]]
negative = [v[:-1] for x, v in enumerate(test.data2) if test.data2[x][2] == 0]
plt.scatter([v[0] for x, v in enumerate(positive)], [v[1] for x, v in enumerate(positive)], c='r', marker='+') 
plt.scatter([v[0] for x, v in enumerate(negative)], [v[1] for x, v in enumerate(negative)], c='b', marker='_')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
# Show the plot 
#plt.show() 

#initialize params for second data set
X = test.data2[:, :-1]
Y = test.data2[:, -1]
ones_X = np.hstack((np.ones((len(X), 1)), X))
initial_theta = np.zeros((len(X[0]) + 1, 1))
#create more features
newX = test.map_feature(X[:, 0].reshape(118, 1), X[:, 1].reshape(118, 1))
new_theta = np.zeros((len(newX[0]), 1))
#compute reg cost
cost, grad = test.cost_function_reg(new_theta, newX, Y, 1)
#print("cost: ", cost[0])
#print("grad:" , grad)

#plot decision boundary around positive examples
x, y = test.find_boundary(np.array(positive)[:, 0], np.array(positive)[:, 1], 7)
plt.plot(x, y, '-k',c='g', lw=2.)

#calculate model accuracy
p = test.predict(new_theta, newX) 
#print("model accuracy: ", p)

plt.show()