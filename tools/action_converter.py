import numpy as np 
import os

from numpy.core.fromnumeric import shape

path="tools/action.csv"
out_path="tools/action_out.csv"
action=np.loadtxt(path,dtype=int,delimiter=",")
with open(out_path,"a+") as file:
    for i in range(shape(action)[0]):
        for j in range(5):
            if j<4:
                file.write(str(list(np.nonzero(action[i,:]))[0][j])+",")
            else:
                file.write(str(list(np.nonzero(action[i,:]))[0][j])+"\n")
                pass
            pass
        pass
    pass



    
# print(shape(action)[0])