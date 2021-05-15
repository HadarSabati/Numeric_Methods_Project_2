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
    final = np.minimum(gaussianFunctions[0], gaussianFunctions[1])
    final = np.minimum(final, gaussianFunctions[2])
    final = np.minimum(final, gaussianFunctions[3])
    final = np.minimum(final, gaussianFunctions[4])
    final = np.minimum(final, gaussianFunctions[5])

    x, y = v
    x = int(x*40 + 400)
    y = int(y*40 + 400)
    if x == 800:
        print ("helloX!")
        x = 799
    if y == 800:
        print ("helloY!")
        y = 799
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

def showSteps(steps):
    for step in steps:
        axes.scatter(step[0], step[1], step[2], color='r')

func = getFunc()
surf1 = axes.plot_surface(X, Y, func, cmap=mycmap, alpha=0.4)
axes.set_title('Gaussian Graph')
fig.colorbar(surf1, ax=axes)  # , shrink=0.5, aspect=10)

## TODO: showSteps with HC and GWO
# plt.show()



## Setting parameters
obj_func = getOutputFromFunc
verbose = True
epoch = 10
pop_size = 50

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
lb1 = [-9.99, -9.99]
ub1 = [9.99, 9.99]

md1 = BaseHC(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[1])
print("done")

# ## 2. When you have same lower bound and upper bound for each parameters, then you can use:
# ##      + int or float: then you need to specify your problem size (number of dimensions)
# problemSize = 10
# lb2 = -5
# ub2 = 10
# md2 = BaseHC(obj_func, lb2, ub2, verbose, epoch, pop_size, problem_size=problemSize)  # Remember the keyword "problem_size"
# best_pos1, best_fit1, list_loss1 = md2.train()
# print(md2.solution[1])
#
# ##      + array: 2 ways
# lb3 = [-5]
# ub3 = [10]
# md3 = BaseHC(obj_func, lb3, ub3, verbose, epoch, pop_size, problem_size=problemSize)  # Remember the keyword "problem_size"
# best_pos1, best_fit1, list_loss1 = md3.train()
# print(md3.solution[1])
#
# lb4 = [-5] * problemSize
# ub4 = [10] * problemSize
# md4 = BaseHC(obj_func, lb4, ub4, verbose, epoch, pop_size)  # No need the keyword "problem_size"
# best_pos1, best_fit1, list_loss1 = md4.train()
# print(md4.solution[1])
#
# # B - Test with algorithm has batch size idea
#
# ## 1. Not using batch size idea
#
# md5 = BaseHC(obj_func, lb4, ub4, verbose, epoch, pop_size)
# best_pos1, best_fit1, list_loss1 = md5.train()
# print(md1.solution[0])
# print(md1.solution[1])
# print(md1.loss_train)
#
# ## 2. Using batch size idea - This algorithm doesn't has batch size idea
#
#
# ## C - Test with different variants of this algorithm
#
# md1 = OriginalHC(obj_func, lb4, ub4, verbose, epoch, pop_size)
# best_pos1, best_fit1, list_loss1 = md1.train()
# print(md1.solution[0])
# print(md1.solution[1])
# print(md1.loss_train)