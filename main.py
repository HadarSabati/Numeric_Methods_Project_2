import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
# matplotlib inline

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.math_based.HC import OriginalHC, BaseHC

from numpy.random import normal
from numpy import sum, mean, exp, array
from mealpy.root import Root

np.random.seed(42)
random.seed(42)

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

final = np.maximum(gaussianFunctions[0], gaussianFunctions[1])
final = np.maximum(final, gaussianFunctions[2])
final = np.maximum(final, gaussianFunctions[3])
final = np.maximum(final, gaussianFunctions[4])
final = np.maximum(final, gaussianFunctions[5])

mycmap = plt.get_cmap('turbo')
fig = plt.figure(figsize=(15, 15))
axes = fig.add_subplot(111, projection='3d')
surf1 = axes.plot_surface(X, Y, final, cmap=mycmap)
axes.set_title('Gaussian Graph')
fig.colorbar(surf1, ax=axes) # , shrink=0.5, aspect=10)


print(final)

plt.show()





#
# ## Setting parameters
# obj_func = F5
# verbose = False
# epoch = 500
# pop_size = 50
#
# # A - Different way to provide lower bound and upper bound. Here are some examples:
#
# ## 1. When you have different lower bound and upper bound for each parameters
# lb1 = [-3, -5, 1]
# ub1 = [5, 10, 100]
#
# md1 = BaseHC(obj_func, lb1, ub1, verbose, epoch, pop_size)
# best_pos1, best_fit1, list_loss1 = md1.train()
# print(md1.solution[1])
#
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
