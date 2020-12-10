from lib.PandapowerTask import *
from lib.GridData import *
import numpy as np


data=GridData()
grid=PandapowerTask()
grid.init_cash(data,env_mode=True,start=0)
print(grid.net.line.sort_index())
print(grid.net.bus.sort_index())

action=np.array([304,472,200,195,287,120,4,8,13,22,29])
s,r,sn=grid.env_step(0,data,action,debug=True)
print(grid.net.res_bus)
print(s)
print(r)
print(sn)
s,r,sn=grid.env_step(1,data,action,debug=True)
print(grid.net.res_bus)
print(s)
print(r)
print(sn)
s,r,sn=grid.env_step(2,data,action,debug=True)
print(grid.net.res_bus)
print(s)
print(r)
print(sn)
s,r,sn=grid.env_step(3,data,action,debug=True)
print(grid.net.res_bus)
print(s)
print(r)
print(sn)