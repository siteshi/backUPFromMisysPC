from bayes_opt import BayesianOptimization
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from mpl_toolkits.mplot3d import Axes3D


def target(x):
    return -1*np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

def multiTarget(x,y):
    return (x+3)**2 + (y -1 )**2 + 1



x = np.linspace(-50000, 10000, 1000)
print type(x)
y = target(x)
print type(y)

z = multiTarget(x,y)

#print type(z)
#print z
#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.plot(x, y, z, label='parametric curve')
#plt.show()


#bo = BayesianOptimization(target,{'x':(-2,10)})
#gp_params = {'corr': 'cubic'}
#bo.maximize(init_points=2, n_iter=2, acq='ucb', kappa=5, **gp_params)
#print (bo.X)
bo = BayesianOptimization(multiTarget,{'x':(-20,100),'y': (-50,100)})
gp_params = {'corr': 'cubic'}
bo.maximize(init_points=2, n_iter=10, acq='ucb',nugget=0.1, kappa=5, **gp_params)
print(bo.res['max'])
print(bo.res['all'])