import numpy as np
import random
import time

class ScenceMaker:
    def __init__(self):
        """
        docstring
        """
        # load the fault scenarios
        ns1 = np.loadtxt("./tools/Nsub1result.csv", dtype=int, delimiter=",")
        ns2 = np.loadtxt("./tools/Nsub2result.csv", dtype=int, delimiter=",")
        event = np.zeros((12864, 37), dtype=int)
        pass

    def make(self, ns1_pr=0.1, ns2_pr=0.8):
        """
        docstring
        """
        time=self.make_time()
        # init the event table
        event = np.zeros((12864, 37), dtype=int)
        #
        fault_rate_ns1 = ns1_pr
        fault_rate_ns2 = ns2_pr
        normal_rate = 1-fault_rate_ns1-fault_rate_ns2
        pr = [normal_rate, fault_rate_ns1, fault_rate_ns2]
        # load the fault scenarios
        ns1 = np.loadtxt("./tools/Nsub1result.csv", dtype=int, delimiter=",")
        ns2 = np.loadtxt("./tools/Nsub2result.csv", dtype=int, delimiter=",")
        for row in range(12864):
            temp_row = np.zeros(37, dtype=int)
            i = np.random.choice([0, 1, 2], p=pr)
            if i == 1:
                temp_row = ns1[random.randint(0, 34), :]
            if i == 2:
                temp_row = ns2[random.randint(0, 458), :]
            event[row, :] = temp_row
            pass
        # save to file
        np.savetxt("./out/res_faultmaker/"+time+"_scene_result.csv",
                   event, fmt="%i", delimiter=",")
        pass
    def make_time(self):
        """
        return a list about current time
        """
        current_time = time.localtime(time.time())
        y = current_time[0]
        mon = current_time[1]
        d = current_time[2]
        h = current_time[3]
        m = current_time[4]
        s = current_time[5]
        res = str(y)+str(mon)+str(d)+"time"+str(h)+"_"+str(m)+"_"+str(s)
        return res

    pass

if __name__ == "__main__":
    mk=ScenceMaker()
    for i in range(500):
        mk.make()
    pass
