import heapq

a = [10, 50, 2, 60, 55, 1]

re2 = map(a.index, heapq.nlargest(3, a))

print(list(re2))