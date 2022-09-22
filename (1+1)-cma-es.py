#
# (1+1)-CMA-ES 
# Author: Kai Tseng (net5566)
#
# The algorithm follows the paper: A Computational Efficient Covariance Matrix Update and a (1+1)-CMA for Evolution Strategies (C. Igel et al., GECCO'06)
# Paper Link: https://christian-igel.github.io/paper/ACECMUaa1p1CMAfES.pdf
#

import numpy as np
import math

#
# Objective function. Here we are going to minimize it
# If you want to approximately compute the maximum, please modify the if statement in the for loop.
#
def objective(x):
    # -0.619 /  -1 / -1
    return (x[0]-3)*(x[0]-3)*(x[0]-4)*(x[0]-5) + (x[1]-1)*(x[1]-2)*(x[1]-3)*(x[1]-4) + (x[2]-1)*(x[2]-2)*(x[2]-3)*(x[2]-4)

# Setting hyperparameters
dim = 3
generations =  500

sigma = 1.0
cov = np.eye(dim)
pc = np.zeros(dim)
psigma = np.zeros(dim)
p_target = 2/11
p_succ = p_target
p_target_ratio = ( p_target / (1 - p_target) )
x_parent =  np.zeros(dim)
p_thresh = 0.44
c_p = 1/12 # learning rate
c_c = 2 / (dim +2)
c_cov = 2 / (pow(dim, 2) + 6)
damping_param = 1+dim/2
check_thresh = True

def updateCov(y):
    global cov, pc, p_succ
    if p_succ < p_thresh:
        pc =  (1-c_c)*pc + math.sqrt(c_c*(2-c_c))*y
        cov = (1-c_cov)*cov + c_cov*np.outer(pc, pc)
    else:
        pc = (1-c_c)*pc
        cov =  (1-c_cov)*cov + c_cov*(np.outer(pc, pc)+c_c*(2-c_c)*cov)

def updateStepSize(lambda_succ):
    global sigma, p_succ
    p_succ = (1-c_p) * p_succ + c_p * lambda_succ
    sigma = sigma * math.exp((p_succ- p_target_ratio*(1-p_succ))/damping_param)

for i in range(generations):
    z = np.random.normal(np.zeros(dim))
    A = np.linalg.cholesky(cov)
    y = np.matmul(A, z)
    x_offspring = x_parent + sigma*y

    if objective(x_offspring) < objective(x_parent):
        updateStepSize(1)
        if i > 400:
            print("Updated! Offspring: ", x_offspring, objective(x_offspring), ", from parent: ", x_parent, objective(x_parent))
        x_parent = x_offspring
        updateCov(y)
    else:
        updateStepSize(0)
