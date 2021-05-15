import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import asarray
from numpy.random import rand
from numpy.random import randn
from mpl_toolkits.mplot3d import axes3d
# matplotlib inline

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.math_based.HC import OriginalHC, BaseHC

from numpy.random import normal
from numpy import sum, mean, exp, array
from mealpy.root import Root
from mpl_toolkits.mplot3d.axes3d import get_test_data

#np.random.seed(42)
#random.seed(42)


## init the variables
r = 0.3
WsArr = [1, r, 0.95 * r, 0.90 * r, 0.85 * r, 0.8 * r]

gaussianFunctions = []
x = np.arange(-10, 10, 0.025)
y = np.arange(-10, 10, 0.025)
X, Y = np.meshgrid(x, y)

for i in range(6):
    yExp = random.uniform(-5, 5)
    xExp = random.uniform(-5, 5)
    yVar = random.uniform(0.06, 0.6)
    xVar = random.uniform(0.06, 0.6)

    # calculating g(x) functions
    Z = WsArr[i]*((np.exp(-(X - xExp) ** 2 / (2 * xVar)))*(np.exp(-(Y - yExp) ** 2 / (2 * yVar))))
    gaussianFunctions.append(Z)

def getFunc():
    final = np.maximum(gaussianFunctions[0], gaussianFunctions[1])
    final = np.maximum(final, gaussianFunctions[2])
    final = np.maximum(final, gaussianFunctions[3])
    final = np.maximum(final, gaussianFunctions[4])
    final = np.maximum(final, gaussianFunctions[5])
    return final

def getOutputFromFunc(v):
    final = np.maximum(gaussianFunctions[0], gaussianFunctions[1])
    final = np.maximum(final, gaussianFunctions[2])
    final = np.maximum(final, gaussianFunctions[3])
    final = np.maximum(final, gaussianFunctions[4])
    final = np.maximum(final, gaussianFunctions[5])

    x, y = v
    x = int(x*40 + 400)
    y = int(y*40 + 400)
    return final[y][x]


mycmap = plt.get_cmap('turbo')
fig = plt.figure(figsize=(15, 15))
axes = fig.add_subplot(111, projection='3d')



# steps = []
# bounds = asarray([[-10.0, 10.0], [-10.0, 10.0]])
# hillClimbing(objective, bounds, 10, 0.7, (5,2))
#
# for step in steps:
#     axes.scatter(step[0], step[1], step[2], color='r')

func = getFunc()
surf1 = axes.plot_surface(X, Y, func, cmap=mycmap, alpha=0.4)
axes.set_title('Gaussian Graph')
fig.colorbar(surf1, ax=axes)  # , shrink=0.5, aspect=10)

plt.show()