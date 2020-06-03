# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:42:12 2020

@author: Surya
Euclidean Distance = 
(sqrt(summation from 1 to n (qi - pi)^2))  
q= (1,3)
p=(2,5)
sqrt((1-2)^2+(3-5)^2)
"""
import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from matplotlib import style 
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5], [7,7], [8,6]]}   #two classes k and r with their features
new_feature = [5,7]

##[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_feature[0], new_feature[1])
##plt.show()

'''
        or it can be implemented as
        
      for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color=i)
            plt.show()
        
        
 '''
#euclidean_distance =sqrt((plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)

#print(euclidean_distance)

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    #knnalgos
    return vote_result


result = k_nearest_neighbors(dataset, new_feature, k=3)
print(result)
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color=result)
plt.show()
