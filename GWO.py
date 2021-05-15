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



np.random.seed(42)
random.seed(42)



def objective(v):
    x, y = v
    x = int(x*40 + 400)
    y = int(y*40 + 400)
    return final[y][x]



    # return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
    # enumerate all dimensions of the point
    for d in range(len(bounds)):
        # check if out of bounds for this dimension
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True


# hill climbing local search algorithm
def hillClimbing(objective, bounds, n_iterations, step_size, start_pt):
    # store the initial point
    solution = start_pt
    # evaluate the initial point
    solution_eval = objective(solution)
    # run the hill climb
    for i in range(n_iterations):
        # take a step
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = solution + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidte_eval = objective(candidate)
        # check if we should keep the new point
        if candidte_eval >= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidte_eval
            print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
            steps.append((solution[0], solution[1], solution_eval))
    return [solution, solution_eval]



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

steps = []
bounds = asarray([[-10.0, 10.0], [-10.0, 10.0]])
hillClimbing(objective, bounds, 10, 0.7, (2,5))

for step in steps:
    axes.scatter(step[0], step[1], step[2], color='r')

# axes.scatter(steps[0][0], steps[0][1], steps[0][2], color='b')
# axes.scatter(steps[1][0], steps[1][1], steps[1][2], color='r')
# axes.scatter(steps[2][0], steps[2][1], steps[2][2], color='g')
# axes.scatter(steps[3][0], steps[3][1], steps[3][2], color='b')
# axes.scatter(steps[4][0], steps[4][1], steps[4][2], color='r')
# axes.scatter(steps[5][0], steps[5][1], steps[5][2], color='g')
# #axes.scatter(steps[6][0], steps[6][1], steps[6][2], color='b')

surf1 = axes.plot_surface(X, Y, final, cmap=mycmap, alpha=0.4)
axes.set_title('Gaussian Graph')
fig.colorbar(surf1, ax=axes) # , shrink=0.5, aspect=10)


plt.show()

