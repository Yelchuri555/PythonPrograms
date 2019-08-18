# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:34:45 2018

@author: Krrish
"""


dancing_list = list(map(int,input().split()))

diff = []

for i in range(len(dancing_list)-1):
    diff.append(dancing_list[i+1]-dancing_list[i])

count = 2

for j in range(len(diff)-1):
    if(diff[j+1]*diff[j]<0):
        count += 1
    
    
print(count)
