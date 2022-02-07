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
                if MI_current > MI_max: # or math.isnan(MI_current)
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
    # random.seed(1) # FIXME: how to set the seed
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

def RM_BM(RM_time):
    # for i in range(config['hp']['components']):
    #     RM_component_index = i
    
    # random.seed(1)
    RM_component_index = random.randint(0, config['hp']['components']-1)
    # print("RM_component_index", RM_component_index)

    # ! donor supply
    donor_supply  = []
    for i in range(len(mask_order[mask_order_index])):
        donor_supply.append([])
        for j in range(len(top_cluster_chromosomes[RM_component_index])):
            donor_supply[i].append(top_cluster_chromosomes[RM_component_index][j][0][mask_order[mask_order_index][i]])
    
    # ! for the same component do RM() with RM_time # TODO: design the flow
    for i in range(RM_time):
        RM_receiver_index = random.randint(0, len(cluster_chromosomes[RM_component_index])-1)
        receiver = cluster_chromosomes[RM_component_index][RM_receiver_index] # ? receiver       
        
        # receiver
        # print("old:" , receiver[1]) # receiver original score
        temp_receiver = deepcopy(receiver[0])
        for i in range(len(mask_order[mask_order_index])):
            mu = np.mean(donor_supply[i])
            sigma = np.std(donor_supply[i])
            donor = random.gauss(mu, sigma)
            temp_receiver[mask_order[mask_order_index][i]] = donor

        update_score = evaluate(temp_receiver, config['op']['task'])
        # print("new:" , update_score)

        if update_score > receiver[1]:
            cluster_chromosomes[RM_component_index][RM_receiver_index][0] = temp_receiver
            cluster_chromosomes[RM_component_index][RM_receiver_index][1] = update_score
            print("RM sucess")
            # ! BM
            BM_component_index = random.randint(0, config['hp']['components']-1)
            while(BM_component_index == RM_component_index):
                BM_component_index = random.randint(0, config['hp']['components']-1)

            BM_receiver_index = random.randint(0, len(cluster_chromosomes[BM_component_index])-1) 
            BM_receiver = cluster_chromosomes[BM_component_index][BM_receiver_index]

            BM_temp_receiver = deepcopy(BM_receiver[0])
            for i in range(len(mask_order[mask_order_index])):
                mu = np.mean(donor_supply[i])
                sigma = np.std(donor_supply[i])
                donor = random.gauss(mu, sigma)
                BM_temp_receiver[mask_order[mask_order_index][i]] = donor

            BM_update_score = evaluate(BM_temp_receiver, config['op']['task'])
            # print("new:" , update_score)

            if BM_update_score > BM_receiver[1]:
                cluster_chromosomes[BM_component_index][BM_receiver_index][0] = BM_temp_receiver
                cluster_chromosomes[BM_component_index][BM_receiver_index][1] = BM_update_score
                print("BM sucess")
                print("---------------------------")

    return cluster_chromosomes




points = np.ndarray((generations, optimizer.population_size, config['hp']['dim']))

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
                    record = [g+1, NFE]
                    writer_object.writerow(record)  
                    f_object.close()
            solutions.append([point,score]) # ! to CMAES
            MI_points.append(point) # ? to calculate MI for buildMask
            scores.append(score)

        mask_order = buildMaskOrder(MI_points, config) # ? to get mask order 
    
        # print(mask_order)
        mask_order_index = 0
        chromosomes = reductDimension(MI_points, mask_order_index) # 0 means the index of mask_order # TODO: the index of mask_order automatically
        # print("generations: {}, mask_order: {}".format(g, mask_order))
        EM = GaussianMixture( n_components = config['hp']['components'])
        EM.fit(chromosomes)
        cluster = EM.predict(chromosomes)   
# """
#         flow:
#         1. 根據 cluster 的值 分成 n 群 # ! ok
#         2. 每群計算 mean 以及 std (從每群中fitness分數較高的) # ! ok
#         3. 同群中隨機挑一個 chromosome 當作 Recevier  # ! ok
#         4. 由 同群 2. sample 出一個 Donor # ! ok
#         5. Donor 給 Recevier 看 fitness 有無變好 # ! ok
#         6. 有變好的話, sample (Donor) 給其他群試試 (其他群先 隨機挑一個 chromosome 當作 Recevier)
#         # ? solutions [ (array([dim0_value, dim1_value, dim2_value]), fitness_score), (array([dim0_value, dim1_value, dim2_value]), fitness_score), ...]
# """
        # solutions.append((point,score)) # ! to CMAES
        # cluster

        cluster_chromosomes = [[] for i in range(config['hp']['components'])]
        for i in range(0, len(solutions)):
            cluster_chromosomes[cluster[i]].append(solutions[i])

        for i in range(config['hp']['components']):
            cluster_chromosomes[i].sort(key = lambda x: x[1])

        top_number = []
        for i in range(config['hp']['components']):
            top_number.append(int((len(cluster_chromosomes[i]) * config['hp']['fitess_top_proportion'])))


        # TODO: RM & BM use sample method
        # ! to calculate mean & std for donor
        top_cluster_chromosomes = [[] for i in range(config['hp']['components'])]
        for i in range(config['hp']['components']):
            top_cluster_chromosomes[i] = cluster_chromosomes[i][:top_number[i]]
     

        cluster_chromosomes = RM_BM(config['hp']['RM_time']) # ! RM
        solutions = []
        for i in range(config['hp']['components']):
            for j in range(len(cluster_chromosomes[i])):
                solutions.append(cluster_chromosomes[i][j])
        # ! Goal: to change the elements of solutions  
        optimizer.tell(solutions) # ! to CMAES 



if not hit:
    print(Fore.YELLOW + Style.BRIGHT + "not hit global optimal")
    with open('record_{}.csv'.format(time.strftime("%Y-%m-%d", time.localtime())), 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        record = [-1, -1]
        writer_object.writerow(record)  
        f_object.close()


 





