
from lib.GridData import *
from lib.PandapowerTask import *

import numpy as np

data=GridData(event="./data/event_normal.csv")
grid=PandapowerTask()
grid.init_output()
grid.init_cache(data,env_mode=True,start=0)

mt_path="./input/mt.csv"
sw_path="./input/sw.csv"

table_mt=np.loadtxt(mt_path,dtype=int,delimiter=",")
table_sw=np.loadtxt(sw_path,dtype=int,delimiter=",")
# temp=list(np.nonzero(table_sw[2,:]))[0].tolist()
# print(temp)

steps=24
for i in range(steps):
    temp=list(np.nonzero(table_sw[i,:]))[0]
    action=np.array([table_mt[i,0],table_mt[i,2],table_mt[i,4],table_mt[i,1],table_mt[i,3],table_mt[i,5],temp[0],temp[1],temp[2],temp[3],temp[4]])
    grid.env_step(i,data,action,debug=True,out=True,log=True)
    print("Proceding... ",round(i*100/(steps-1),2))
    print("%")
    pass