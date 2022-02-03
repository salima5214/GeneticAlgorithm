import numpy as np


a = np.array([[1, 2], [4], [5, 6]], dtype=object)
print(a)
index = [0, 2]

new_a = np.delete(a, index, axis = 0)


print(new_a)
new_b = np.array([4, 5, 6], dtype=object)
new_c = np.row_stack((new_a, new_b), axis = 0)

print(new_c)
