import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from logisticRegression import logisticRegression

test = logisticRegression()
#original data plot
positive = [v[:-1] for x, v in enumerate(test.data) if test.data[x][2]]
negative = [v[:-1] for x, v in enumerate(test.data) if test.data[x][2] == 0]
plt.scatter([v[0] for x, v in enumerate(positive)], [v[1] for x, v in enumerate(positive)], c='r', marker='+') 
plt.scatter([v[0] for x, v in enumerate(negative)], [v[1] for x, v in enumerate(negative)], c='b', marker='_')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
# Show the plot 
plt.show() 

