# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:06:19 2018

@author: Krrish
"""

import pandas as pd
import numpy as np

machine_details = open("machineId.txt")

content = machine_details.readlines()

print(len(content))

details = {"MachineID":[],"snap1":[],"snap2":[]}

snap_count = 0
for i in range(0,len(content)):
    print(i)
    for j in content[i].split():
        if(j.startswith("ami-")):
            details['MachineID'].append(j)
        elif(j.startswith("snap-")):
            if(snap_count%2==0):
                details['snap1'].append(j)
                snap_count += 1
            else:
                details['snap2'].append(j)
                snap_count += 1

print(details)
                
                
details = pd.DataFrame(details) 
details.to_csv("Machine_details.csv",index = False)              
                
    


#name = "ramakrishna"
#
#print(name.startswith("ram"))

machine_details.close()