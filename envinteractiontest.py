from lib.PandapowerTask import *
from lib.GridData import *
import numpy as np


data=GridData()
grid=PandapowerTask()
grid.init_cache(data,env_mode=True,start=0)
print(grid.net.line.sort_index())
print(grid.net.bus.sort_index())

action=np.array([1,0,0,15,27,10,27,32,33,34,35
])
s,r,sn,f=grid.env_step(91,data,action,debug=True)
print(grid.net.res_bus)
print(s)
print(r)
print(sn)
print(f)
# s,r,sn,f=grid.env_step(1,data,action,debug=True)
# print(grid.net.res_bus)
# print(s)
# print(r)
# print(sn)
# print(f)
# s,r,sn,f=grid.env_step(2,data,action,debug=True)
# print(grid.net.res_bus)
# print(s)
# print(r)
# print(sn)
# print(f)
# s,r,sn,f=grid.env_step(3,data,action,debug=True)
# print(grid.net.res_bus)
# print(s)
# print(r)
# print(sn)
# print(f)