from my_cmaes import CMA
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from sklearn.mixture import GaussianMixture
import heapq
from copy import deepcopy

NFE = 0

def evaluate(point):
    global NFE
    NFE += 1
    return (100*((point[2]-(point[1]**2))**2)+(1-point[1])**2)+(100*((point[1]-(point[0]**2))**2)+(1-point[0])**2)

optimizer = CMA(mean=np.array([2.5,2.5,2.5]), bounds=np.array([[0,5],[0,5],[0,5]]), sigma=0.5, n_max_resampling=1)
generations = 16

sqrt = int(np.sqrt(generations))
fig, axs = plt.subplots(sqrt, sqrt, num="CMA-ES", sharex=True, sharey=True)
points = np.ndarray((generations, optimizer.population_size, 3))
points_1 = np.ndarray((generations, optimizer.population_size, 3))
points_2 = np.ndarray((generations, optimizer.population_size, 3))
hit = False

for g in range(generations):
    if hit: break
    if not hit:
        solutions = []
        scores = []
        chromosomes = []
        for i in range(optimizer.population_size):
            point = optimizer.ask() # sample next chromosome   
            points[g, i] = point # record
            score = evaluate(point) # evaluate fitness
            if score < 0.001 and not hit:
                print("hit global optimal")
                print("point = {}".format(point))
                print("generations = {}".format(g+1))
                print("NFE = {}".format(NFE))
                hit = True     
            solutions.append((point,score))
            scores.append(score)
            chromosomes.append((point[0], point[1]))
        
        np.random.seed(1)
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

 
# target = np.array([1, 1, 1])
for i in range(generations):
    ax = axs[i//sqrt, i%sqrt]
    ax.scatter(*zip(*points_1[i]), c="b")
    ax.scatter(*zip(*points_2[i]), c="r")
    # ax.scatter(*target,c="r")
    ax.set_xlim([0,5])
    ax.set_ylim([0,5])

plt.savefig('cmaes_test.png')


