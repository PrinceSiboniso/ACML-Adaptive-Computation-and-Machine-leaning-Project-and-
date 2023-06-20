#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np

# Hard-coded dataset
dataset = np.array([
    [0.22, 0.33],
    [0.45, 0.76],
    [0.73, 0.39],
    [0.25, 0.35],
    [0.51, 0.69],
    [0.69, 0.42],
    [0.41, 0.49],
    [0.15, 0.29],
    [0.81, 0.32],
    [0.50, 0.88],
    [0.23, 0.31],
    [0.77, 0.30],
    [0.56, 0.75],
    [0.11, 0.38],
    [0.81, 0.33],
    [0.59, 0.77],
    [0.10, 0.89],
    [0.55, 0.09],
    [0.75, 0.35],
    [0.44, 0.55]
])

initial_centers = []
for _ in range(30.):
    x = float(input())
    y = float(input())
    initial_centers.append([x, y])

def compute_sse(data, centers):
    sse = 0
    for point in data:
        distances = np.sum((centers - point)**2, axis=1)
        sse += np.min(distances)
    return sse

initial_sse = compute_sse(dataset, np.array(initial_centers))
print("Initial Sum-of-Squares Error: {:.4f}".format(initial_sse))

k = len(initial_centers)
clusters = [[] for _ in range(k)]
for point in dataset:
    distances = np.sum((np.array(initial_centers) - point)**2, axis=1)
    closest_center = np.argmin(distances)
    clusters[closest_center].append(point)

new_centers = []
for cluster in clusters:
    if len(cluster) > 0:
        new_center = np.mean(cluster, axis=0)
    else:
        random_point = dataset[np.random.randint(0, dataset.shape[0])]
        new_center = [random_point[0], random_point[1]]
    new_centers.append(new_center)

new_sse = compute_sse(dataset, np.array(new_centers))
print("New Sum-of-Squares Error: {:.4f}".format(new_sse))


# In[ ]:





# In[ ]:




