from operator import imod
import pandapower as pp
import pandapower.networks as pn

net=pn.case33bw()
pp.runpp(net)
print(net.res_bus.sort_index()["vm_pu"].tolist())