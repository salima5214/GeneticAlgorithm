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


def calculateMaskCoefficient(MI_points, mask_1, mask_2):
    r = 0
    for i in range(len(mask_1)):
        for j in range(len(mask_2)):
            r += np.corrcoef(MI_points[:,mask_1[i]], MI_points[:,mask_2[j]])[0, 1]
    return r

def buildMask(MI_points):
    masks = np.array([[index] for index in range(len(MI_points[0]))])
    MI_points = np.array(MI_points)
    MI_max = -float('inf')
    mask_order = []

    while(len(masks) >= 2):
        MI_max = -float('inf')
        for mask_index1 in range(0, len(masks)-1):
            for mask_index2 in range(mask_index1+1, len(masks)):
                r = calculateMaskCoefficient(MI_points, masks[mask_index1], masks[mask_index2])      
                MI_current = math.log(1/((1-(r*r))**0.5))
                if MI_current > MI_max or math.isnan(MI_current):
                    MI_max = MI_current
                    remove_mask_index = np.array([mask_index1, mask_index2])
                    new_mask = np.hstack((masks[mask_index1],  masks[mask_index2]))

        print("******************************************")
        print(masks, "masks_pre")
        print(mask_order, "mask_order_pre")
        print(remove_mask_index, "remove_mask_index") 
        print(new_mask, "new_mask")
        masks = np.delete(masks, remove_mask_index, axis = 0)
        masks = masks.tolist()
        masks.append(new_mask.tolist())
        mask_order.append(new_mask.tolist())
        masks = np.array(masks, dtype=object)
        print(masks, "masks_post")
        print(mask_order, "mask_order_post")
        print("******************************************")
    

        
points = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))
points_1 = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))
points_2 = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))



for g in range(generations):
    if hit: break
    if not hit:
        solutions = []
        scores = []
        MI_points = []
        chromosomes = []
        for i in range(optimizer.population_size):
            point = optimizer.ask() # ! sample next chromosome   
            points[g, i] = point # record
            # print(point)
            score = evaluate(point, config['op']['task']) # evaluate fitness
            if score < 0.001 and not hit:
                print("hit global optimal")
                print("point = {}".format(point))
                print("generations = {}".format(g+1))
                print("NFE = {}".format(NFE))
                hit = True     
            solutions.append((point,score))
            MI_points.append(point) # ! to calculate MI for buildMask
            scores.append(score)
            chromosomes.append((point[0], point[1]))

        buildMask(MI_points) # ! to get mask order
        EM = GaussianMixture( n_components = 2)
        EM.fit(chromosomes)
        cluster = EM.predict(chromosomes)
        # print(cluster)

        min_index_list = list(map(scores.index, heapq.nsmallest(100, scores)))
        count_0 = 0
        count_1 = 0
        chromosomes_0 = []
        chromosomes_1 = []

        for i in range(100):
            if cluster[min_index_list[i]] == 0 and count_0 <= 20:
                chromosomes_0.append((solutions[min_index_list[i]], min_index_list[i]))
                count_0 += 1
            elif cluster[min_index_list[i]] == 1 and count_1 <= 20:
                chromosomes_1.append((solutions[min_index_list[i]], min_index_list[i]))
                count_1 += 1
            elif count_0 <= 20 or count_1 <= 20:
                continue
            else:
                break


        for i in range(0):
            # set_trace()
            if g <= 5 and chromosomes_0[i][0][1] < chromosomes_1[i][0][1]:
                temp = deepcopy(solutions[chromosomes_1[i][1]])
                temp[0][0] = solutions[chromosomes_0[i][1]][0][0]
                temp[0][1] = solutions[chromosomes_0[i][1]][0][1]
                # if evaluate(temp[0]) < evaluate(solutions[chromosomes_1[i][1]][0]):
                if True:
                    print("+++++++++++++++++++")
                    solutions[chromosomes_1[i][1]][0][0] = solutions[chromosomes_0[i][1]][0][0]
                    solutions[chromosomes_1[i][1]][0][1] = solutions[chromosomes_0[i][1]][0][1]
            elif g <= 5 and chromosomes_0[i][0][1] > chromosomes_1[i][0][1]:
                temp = deepcopy(solutions[chromosomes_0[i][1]])
                temp[0][0] = solutions[chromosomes_1[i][1]][0][0]
                temp[0][1] = solutions[chromosomes_1[i][1]][0][1]
                # if evaluate(temp[0]) < evaluate(solutions[chromosomes_0[i][1]][0]):
                if True:
                    print("#####################")
                    solutions[chromosomes_0[i][1]][0][0] = solutions[chromosomes_1[i][1]][0][0]
                    solutions[chromosomes_0[i][1]][0][1] = solutions[chromosomes_1[i][1]][0][1]


        index_1 = 0
        index_2 = 0
        for i in range(optimizer.population_size):
            if cluster[i] == 0:
                points_1[g, index_1] = points[g, i]
                index_1 += 1
            else:
                points_2[g, index_2] = points[g, i]
                index_2 += 1

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
