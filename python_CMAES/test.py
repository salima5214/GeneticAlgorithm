import numpy as np
import random

import colorama
from colorama import Fore
from colorama import Style

colorama.init()
print(Fore.BLUE + Style.BRIGHT + "This is the color of the sky")
print(Fore.GREEN + "This is the color of grass")
print(Fore.MAGENTA + Style.BRIGHT + "This is a dimmer version of the sky")
print(Fore.YELLOW + Style.BRIGHT + "This is a dimmer version of the sky")
print(Fore.WHITE + Style.BRIGHT + "This is a dimmer version of the sky")
print(Fore.CYAN + Style.BRIGHT + "This is a dimmer version of the sky")
print(Fore.BLACK + Style.BRIGHT + "This is a dimmer version of the sky")

print(Fore.RED + Style.DIM + "This is a dimmer version of the sky")
print(Fore.YELLOW + "This is the color of the sun" + Style.RESET_ALL)


# !
# MI_points = [1, 2 ,3 ,4]
# init_index = random.randint(0, len(MI_points)-1)
# in_mask = [0]
# out_mask = [index for index in range(0, len(MI_points)) if index is not init_index]
# mask_order = []
# mask_order.append(in_mask)
# print("init in_mask = {}, mask_order = {}".format(in_mask, mask_order))
# mask_order.append([0,1])
# print("init in_mask = {}, mask_order = {}".format(in_mask, mask_order))

# !
# my_set = { i for i in range(5)}
# my_set.remove(2)
# print(my_set)

#!
# import random

# print(random.randint(0,1))


# ! 
# chromosomes = []

# chromosomes.append([])

# print(chromosomes)

# chromosomes[0].append(3)
# print(chromosomes[0])
# chromosomes[0].append(5)
# print(chromosomes)

# ! 
# a = np.array([[1, 2], [4], [5, 6]], dtype=object)
# print(a)
# index = [0, 2]

# new_a = np.delete(a, index, axis = 0)


# print(new_a)
# new_b = np.array([4, 5, 6], dtype=object)
# new_c = np.row_stack((new_a, new_b), axis = 0)

# print(new_c)
