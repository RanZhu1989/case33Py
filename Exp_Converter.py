
from lib.GridData import *
from lib.PandapowerTask import *

import numpy as np

data=GridData()
grid=PandapowerTask()
grid.init_output()
grid.init_cache(data,env_mode=True,start=0)



action=np.array([304,472,200,195,287,120,1,8,13,22,29])
grid.env_step(0,data,action,debug=True,out=True,log=True)