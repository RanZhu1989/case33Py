from lib.PandapowerTask import *
from lib.GridData import *
import numpy as np


data=GridData()
panda=PandapowerTask()
print(panda.net.load)
action=np.array([750,125,375,32,39,25,1,8,23,28,32])
s,r,sn=panda.env_step(3,data,action)
print(panda.net.res_bus["vm_pu"])
print(s)
print(r)
print(sn)

