
from lib.GridData import GridData
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as pt
import numpy as np
import random


class FaultMaker():
    '''
    Generate n-1 n-2 fault scenarios by pandapower & network_X

    '''

    def __init__(self):
        """
        Create a standard model 
        - default case = case33BW 

        """

        self.prim_net = pn.case33bw()
        # set the five tie-switches "in service"
        self.prim_net.line = self.prim_net.line.sort_index()
        for i in range(32, 37):
            self.prim_net.line["in_service"][i] = True
            pass

        pass

    def make_fault(self, mode="N-1", island_allow=False, island_bus=False, loop=1e6):
        """
        make fault sense of lines, buses ...

        # searching loop is set to 1e6 as defult   

        - mode = "N-1" ; "N-2" ; "LPHI"
            * "LPHI" mode is an interface for meteorological disasters *

        - island_allow = False : line[0]=(1,2) cant be distoried
          island_allow = True  : the system could be operation in island mode as microgrid

        - island_bus = False : all buses could be re-energized by reconfiguration
          island_bus = True  : some buses cannot be recovered by reconfiguration

        #TODO : Current function =  mode = "N-1" & "N-2" ; island_allow = False ; island_bus = False  

        """
        # reset net
        self.reflash_net()
        res_sense = np.zeros((loop,37), dtype=int)
        if mode == "N-1":
            s=0
            for i in range(loop):
                self.reflash_net()
                faultlist = random.randint(1, 36)
                if island_allow == False:
                    if island_bus == False:
                        self.prim_net.line["in_service"][faultlist] = False
                        if len(pt.unsupplied_buses(self.prim_net)) == 0:
                            res_sense[s,faultlist] = 1
                            s+=1
                            pass
                        pass
                    pass
                pass
            # remove duplicate lines
            res_sense=np.unique(res_sense,axis=0)
            res_sense=np.delete(res_sense,[-1],axis=0)
            np.savetxt("./out/res_faultmaker/result.csv", res_sense,fmt="%i", delimiter=",")
            pass

        if mode == "N-2":
            s=0
            for i in range(loop):
                self.reflash_net()
                faultlist = []
                while(len(faultlist) < 2):
                    x = random.randint(1, 36)
                    if x not in faultlist:
                        faultlist.append(x)
                        pass
                    pass
                if island_allow == False:
                    if island_bus == False:
                        self.prim_net.line["in_service"][faultlist[0]] = False
                        self.prim_net.line["in_service"][faultlist[1]] = False
                        if len(pt.unsupplied_buses(self.prim_net)) == 0:
                            res_sense[s,faultlist] = 1
                            s+=1
                            pass
                        pass
                    pass
                pass
            # remove duplicate lines
            res_sense=np.unique(res_sense,axis=0)
            res_sense=np.delete(res_sense,[-1],axis=0)
            np.savetxt("./out/res_faultmaker/result.csv", res_sense,fmt="%i", delimiter=",")
            pass
        pass

    def reflash_net(self):
        """
        docstring
        """
        self.prim_net = pn.case33bw()
        # set the five tie-switches "in service"
        self.prim_net.line = self.prim_net.line.sort_index()
        for i in range(32, 37):
            self.prim_net.line["in_service"][i] = True
            pass
        pass
    
    def check_radial(self,data:GridData):
        """
        docstring
        """
        pass
    
    
    pass


if __name__ == "__main__":
    mk = FaultMaker()
    print(mk.prim_net.line)
    pass
