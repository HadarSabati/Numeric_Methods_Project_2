#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:29, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.math_based.HC import OriginalHC, BaseHC
import initFunction

## Setting parameters
obj_func = getOutputFromFunc()
verbose = True
epoch = 10
pop_size = 50

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
lb1 = [-10, -10]
ub1 = [10, 10]

md1 = BaseHC(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[1])

## 2. When you have same lower bound and upper bound for each parameters, then you can use:
##      + int or float: then you need to specify your problem size (number of dimensions)
problemSize = 10
lb2 = -5
ub2 = 10
md2 = BaseHC(obj_func, lb2, ub2, verbose, epoch, pop_size, problem_size=problemSize)  # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md2.train()
print(md2.solution[1])

##      + array: 2 ways
lb3 = [-5]
ub3 = [10]
md3 = BaseHC(obj_func, lb3, ub3, verbose, epoch, pop_size, problem_size=problemSize)  # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md3.train()
print(md3.solution[1])

lb4 = [-5] * problemSize
ub4 = [10] * problemSize
md4 = BaseHC(obj_func, lb4, ub4, verbose, epoch, pop_size)  # No need the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md4.train()
print(md4.solution[1])

# B - Test with algorithm has batch size idea

## 1. Not using batch size idea

md5 = BaseHC(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md5.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

## 2. Using batch size idea - This algorithm doesn't has batch size idea


## C - Test with different variants of this algorithm

md1 = OriginalHC(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)