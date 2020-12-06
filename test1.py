from numpy.core.records import array
import pandapower as pp
import pandapower.networks as pn
import numpy as np
net= pn.case33bw()
pp.runpp(net)
for i in range(10):
    print(str(np.array(net.res_bus.sort_index().drop(0,0)["p_mw"])[i]))

