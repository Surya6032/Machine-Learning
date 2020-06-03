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
from collections import Counter
import pandas as pd
import random

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
    vote_result = Counter(votes).most_common(1)[0][0]
    #knnalgos
    return vote_result

df= pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)                   #-9999 represent outlier
df.drop(['id'], 1 , inplace=True)
full_data = df.astype(float).values.tolist()            #converting everything in data to float
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}                               
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)) : ]               #test will be from last 20% of data

#Populated the dictionaries 
for i in train_data:
     train_set[i[-1]].append(i[:-1])
    
for i in test_data:
     test_set[i[-1]].append(i[:-1])
     
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct +=1
        total += 1
        
print('Accuracy:', correct/total)