from my_cmaes import CMA
import numpy as np
import matplotlib.pyplot as plt


NFE = 0

def evaluate(point):
    global NFE
    NFE += 1
    return 100*((point[1]-(point[0]**2))**2)+(1-point[0])**2

optimizer = CMA(mean=np.array([0.5,0.5]), bounds=np.array([[0,1],[0,1]]), sigma=0.5, n_max_resampling=1)
generations = 5

sqrt = int(np.sqrt(generations))
fig, axs = plt.subplots(sqrt, sqrt, num="CMA-ES", sharex=True, sharey=True)
points = np.ndarray((generations, optimizer.population_size, 2))


hit = False

for g in range(generations):

    if hit:
        break 

    if not hit:
        solutions = []
        for i in range(optimizer.population_size):
            point = optimizer.ask()
            points[g, i] = point
            score = evaluate(point)
            
            if score < 0.001 and not hit:
                print("hit global optimal")
                print("generations = {}".format(g+1))
                print("NFE = {}".format(NFE))
                hit = True     

            solutions.append((point,score))
        optimizer.tell(solutions)

if not hit:
    print("not hit global optimal")
    print("generations = {}".format(g+1))
    print("NFE = {}".format(NFE))

 




