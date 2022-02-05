from my_cmaes import CMA
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from sklearn.mixture import GaussianMixture
import heapq
from copy import deepcopy
import math
import yaml
from scipy.stats import pearsonr


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)

# hyperparameter settings
np.random.seed(config['hp']['np_seed'])
NFE = 0
dim = config['hp']['dim']
generations = config['hp']['generations']
hit = False
my_bound_min = config['hp']['bound_min']
my_bound_max = config['hp']['bound_max']
my_bounds = np.array([[my_bound_min, my_bound_max]]*dim)

np.ones(dim)
init_mean = config['hp']['init_mean']
my_mean = np.ones(dim)*init_mean

optimizer = CMA(mean=my_mean, bounds=my_bounds, sigma=0.5, n_max_resampling=1)

# print information of this experiment
def printInfo(task):
    if task == 0: #  0: Sphere      
        print("Sphere")
    elif task == 1: #  1: Rosenbrock
        print("Rosenbrock")
    elif task == 2: #  2: Rastrigin
        print("Rastrigin")
    elif task == 3: #  3: Michalewicz
        print("Michalewicz")
    print("dim = {} ".format(config['hp']['dim']))

task = config['op']['task']
printInfo(task)

def evaluate(point, task):
    global NFE
    NFE += 1
    fitness = 0
    if task == 0: #  0: Sphere      
        for i in range(0, len(point)):
            fitness += (point[i]**2)
        return fitness
    elif task == 1: #  1: Rosenbrock
        for i in range(0, len(point)-1):
            fitness += (100*((point[i+1]-(point[i]**2))**2)+(1-point[i])**2)
        return fitness
    elif task == 2: #  2: Rastrigin
        for i in range(0, len(point)):
            fitness += ((point[i]**2)-10*math.cos(2*math.pi*point[i]))
        fitness += 10*len(point)
        return fitness
    elif task == 3: #  3: Michalewicz
        for i in range(0, len(point)):
            fitness += (-1*(math.sin(point[i]))*math.pow(math.sin((i+1)*point[i]*point[i]/(math.pi)),20))
        return fitness





points = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))
points_1 = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))
points_2 = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))


for g in range(generations):
    if hit: break
    if not hit:
        solutions = []
        scores = []
        MI_points = []

        for i in range(optimizer.population_size):
            point = optimizer.ask() # ! sample next chromosome   
            points[g, i] = point # record

            score = evaluate(point, config['op']['task']) # evaluate fitness
            if score < 0.001 and not hit:
                print("hit global optimal")
                print("point = {}".format(point))
                print("generations = {}".format(g+1))
                print("NFE = {}".format(NFE))
                hit = True     
            solutions.append((point,score))
        optimizer.tell(solutions)

if not hit:
    print("not not not not not hit global optimal")
    # print("generations = {}".format(g+1))
    # print("NFE = {}".format(NFE))

 




# sqrt = int(np.sqrt(generations))
# fig, axs = plt.subplots(sqrt, sqrt, num="CMA-ES", sharex=True, sharey=True)

# target = np.array([1, 1, 1])
# for i in range(generations):
#     ax = axs[i//sqrt, i%sqrt]
#     ax.scatter(*zip(*points_1[i]), c="b")
#     ax.scatter(*zip(*points_2[i]), c="r")
#     # ax.scatter(*target,c="r")
#     ax.set_xlim([0,5])
#     ax.set_ylim([0,5])

# plt.savefig('cmaes_test.png')
