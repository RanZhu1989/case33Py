from seaborn.axisgrid import Grid
from lib.PandapowerTask import *
from lib.GridData import *
import numpy as np


data=GridData()
panda=PandapowerTask()
action=np.array([750,125,375,35,39,25,1,8,23,28,32])
s,r,sn=panda.env_step(3,data,action)
print(s)
print(r)
print(sn)