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
import time
from csv import writer
import argparse
import random
import colorama
from colorama import Fore
from colorama import Style

# parser = argparse.ArgumentParser()
# parser.add_argument("top", help="top number", type=int)
# args = parser.parse_args()

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
def printInfo(config):
    if config['op']['task'] == 0: #  0: Sphere      
        print(Fore.MAGENTA + Style.BRIGHT + "Problem: Sphere")
    elif config['op']['task'] == 1: #  1: Rosenbrock
        print(Fore.MAGENTA + Style.BRIGHT + "Problem: Rosenbrock")
    elif config['op']['task'] == 2: #  2: Rastrigin
        print(Fore.MAGENTA + Style.BRIGHT + "Problem: Rastrigin")
    elif config['op']['task'] == 3: #  3: Michalewicz
        print(Fore.MAGENTA + Style.BRIGHT + "Problem: Michalewicz")

    if config['op']['mask_order'] == 0: #  0: linkage tree order    
        print(Fore.CYAN + Style.BRIGHT + "The way of mask order: linkage tree order")
    elif config['op']['mask_order'] == 1: #  1: incremental linkage set
        print(Fore.CYAN + Style.BRIGHT + "The way of mask order: incremental linkage set")
    
    print(Fore.BLUE + Style.BRIGHT + "dimension: {} ".format(config['hp']['dim']))

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

def calculateMI(MI_points, mask_1, mask_2):
    MI = 0
    count = 0
    for i in range(len(mask_1)):
        for j in range(len(mask_2)):
            r = np.corrcoef(MI_points[:,mask_1[i]], MI_points[:,mask_2[j]])[0, 1]
            MI += math.log(1/((1-(r*r))**0.5))
            count += 1
    MI = MI / count # FIXME: the way to calculate MI for two masks
    return MI

def byLinkageTree(MI_points): # ! Linkage Tree
    masks = np.array([[index] for index in range(len(MI_points[0]))])
    MI_points = np.array(MI_points)
    mask_order = []

    while(len(masks) >= 2):
        MI_max = -float('inf')
        for mask_index1 in range(0, len(masks)-1):
            for mask_index2 in range(mask_index1+1, len(masks)): 
                MI_current = calculateMI(MI_points, masks[mask_index1], masks[mask_index2])
                if MI_current > MI_max or math.isnan(MI_current):
                    MI_max = MI_current
                    remove_mask_index = np.array([mask_index1, mask_index2])
                    new_mask = np.hstack((masks[mask_index1],  masks[mask_index2]))

        # print("******************************************")
        # print(masks, "masks_pre")
        # print(mask_order, "mask_order_pre")
        # print(remove_mask_index, "remove_mask_index") 
        # print(new_mask, "new_mask")
        masks = np.delete(masks, remove_mask_index, axis = 0)
        masks = masks.tolist()
        masks.append(new_mask.tolist())
        mask_order.append(new_mask.tolist())
        masks = np.array(masks, dtype=object)
        # print(masks, "masks_post")
        # print(mask_order, "mask_order_post")
        # print("******************************************")
    return mask_order

def byIncrementalLinkageSet(MI_points): # ! ILS
    random.seed(1)
    init_index = random.randint(0, len(MI_points[0])-1)
    # print("init_index: ", init_index)
    in_mask = [init_index]
    out_mask = [index for index in range(0, len(MI_points[0])) if index is not init_index]
    # mask_order = []
    # mask_order.append(in_mask)
    for i in range(len(MI_points[0])-1):
        MI_max = -float('inf')
        for out_mask_index in out_mask:
            for in_mask_index in in_mask: 
                MI_current = calculateMI(np.array(MI_points), np.array([out_mask_index]), np.array(in_mask))
                if MI_current > MI_max or math.isnan(MI_current):
                    MI_max = MI_current
                    remove_mask_index = out_mask_index

        out_mask.remove(remove_mask_index)
        in_mask.append(remove_mask_index)
        # mask_order.append(in_mask)
        # print("i = {} , in_mask = {}, out_mask = {}".format(i, in_mask, out_mask))

    mask_order = []

    for i in range(1, len(in_mask)):
        mask = []
        for j in range(i+1):
            mask.append(in_mask[j])
        mask_order.append(mask)   
    # print("mask_order = {}".format(mask_order))
    return mask_order

def buildMaskOrder(MI_points, config):
    if config['op']['mask_order'] == 0: #  0: linkage tree order    
        return byLinkageTree(MI_points)
    elif config['op']['mask_order'] == 1: #  1: incremental linkage set
        return byIncrementalLinkageSet(MI_points)

def reductDimension(MI_points, mask_order_index):
    chromosomes = []
    for i in range(0, len(MI_points)):
        for j in range(0, len(mask_order[mask_order_index])):
            if j == 0:
                chromosomes.append([])
            chromosomes[i].append(MI_points[i][mask_order[mask_order_index][j]])
    return chromosomes

points = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))
points_1 = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))
points_2 = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))

printInfo(config)
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
            if score < 0.001 and not hit: # FIXME: check global optima value always = 0?
                print(Fore.RED + Style.BRIGHT + "hit global optimal")
                print(Fore.WHITE + Style.BRIGHT + "hit point: {}".format(point))
                print("generations: {}".format(g+1))
                print(Fore.YELLOW + Style.BRIGHT + "NFE: {}".format(NFE))
                hit = True
                with open('record_{}.csv'.format(time.strftime("%Y-%m-%d", time.localtime())), 'a', newline='') as f_object:  
                    writer_object = writer(f_object)
                    record = [0, NFE]
                    writer_object.writerow(record)  
                    f_object.close()
            solutions.append((point,score))
            MI_points.append(point) # ! to calculate MI for buildMask
            scores.append(score)

        mask_order = buildMaskOrder(MI_points, config) # ! to get mask order 
    
        # print(mask_order)
        chromosomes = reductDimension(MI_points, 0) # 0 means the index of mask_order # TODO: the index of mask_order automatically
        # print("generations: {}, mask_order: {}".format(g, mask_order))
        EM = GaussianMixture( n_components = 2)
        EM.fit(chromosomes)
        cluster = EM.predict(chromosomes)   

#########################################################################################
#########################################################################################
#########################################################################################

        min_index_list = list(map(scores.index, heapq.nsmallest(100, scores)))
        count_0 = 0
        count_1 = 0
        chromosomes_0 = []
        chromosomes_1 = []
        # TODO: cluster automatically
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

        # TODO: RM & BM use sample method
        for i in range(5): 
            # set_trace()
            if g <= 3 and chromosomes_0[i][0][1] < chromosomes_1[i][0][1]:
                temp = deepcopy(solutions[chromosomes_1[i][1]])
                temp[0][mask_order[0][0]] = solutions[chromosomes_0[i][1]][0][mask_order[0][0]]
                temp[0][mask_order[0][1]] = solutions[chromosomes_0[i][1]][0][mask_order[0][1]]
                # if evaluate(temp[0], config['op']['task']) < evaluate(solutions[chromosomes_1[i][1]][0], config['op']['task']):
                if True:
                    # print("+++++++++++++++++++")
                    solutions[chromosomes_1[i][1]][0][mask_order[0][0]] = solutions[chromosomes_0[i][1]][0][mask_order[0][0]]
                    solutions[chromosomes_1[i][1]][0][mask_order[0][1]] = solutions[chromosomes_0[i][1]][0][mask_order[0][1]]
            elif g <= 3 and chromosomes_0[i][0][1] > chromosomes_1[i][0][1]:
                temp = deepcopy(solutions[chromosomes_0[i][1]])
                temp[0][mask_order[0][0]] = solutions[chromosomes_1[i][1]][0][mask_order[0][0]]
                temp[0][mask_order[0][1]] = solutions[chromosomes_1[i][1]][0][mask_order[0][1]]
                # if evaluate(temp[0], config['op']['task']) < evaluate(solutions[chromosomes_0[i][1]][0], config['op']['task']):
                if True:
                    # print("#####################")
                    solutions[chromosomes_0[i][1]][0][mask_order[0][0]] = solutions[chromosomes_1[i][1]][0][mask_order[0][0]]
                    solutions[chromosomes_0[i][1]][0][mask_order[0][1]] = solutions[chromosomes_1[i][1]][0][mask_order[0][1]]
        optimizer.tell(solutions)

if not hit:
    print(Fore.YELLOW + Style.BRIGHT + "not hit global optimal")
    with open('record_{}.csv'.format(time.strftime("%Y-%m-%d", time.localtime())), 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        record = [-1, -1]
        writer_object.writerow(record)  
        f_object.close()
    # print("generations = {}".format(g+1))
    # print("NFE = {}".format(NFE))

 





