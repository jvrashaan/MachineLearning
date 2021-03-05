import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from linearRegressionMultiVar import linear_regression_multi_var

test = linear_regression_multi_var()

#plot the data
plt.plot(test.data2, test.labels, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

#get cost and grad 
theta = np.ones((2, 1))
cost = test.compute_cost_multi_reg(test.data2, test.labels, theta, 1)
print('Cost at theta = [1 ; 1]: ', cost)

