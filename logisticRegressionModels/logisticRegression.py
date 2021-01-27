import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class logisticRegression():
    #Given a data set of home features and home prices this class will predict the price of homes
    # given the features home size and number of bedrooms.
    def __init__(self):
        with open('ex2data1.txt') as f:
            data = f.read()
        
        self.data = np.array(re.split(',|\n', data)).reshape((100, 3)).astype(np.float)

    