import numpy as np
import random


class ScenceMaker:
    def __init__(self):
        """
        docstring
        """
        # load the fault scenarios
        ns1 = np.loadtxt("./tools/Nsub1result.csv", dtype=int, delimiter=",")
        ns2 = np.loadtxt("./tools/Nsub2result.csv", dtype=int, delimiter=",")
        event = np.zeros((12865, 37), dtype=int)
        pass

    def make(self, ns1_pr=0.1, ns2_pr=0.01):
        """
        docstring
        """
        # init the event table
        event = np.zeros((12865, 37), dtype=int)
        #
        fault_rate_ns1 = ns1_pr
        fault_rate_ns2 = ns2_pr
        normal_rate = 1-fault_rate_ns1-fault_rate_ns2
        pr = [normal_rate, fault_rate_ns1, fault_rate_ns2]
        # load the fault scenarios
        ns1 = np.loadtxt("./tools/Nsub1result.csv", dtype=int, delimiter=",")
        ns2 = np.loadtxt("./tools/Nsub2result.csv", dtype=int, delimiter=",")
        for row in range(12865):
            temp_row = np.zeros(37, dtype=int)
            i = np.random.choice([0, 1, 2], p=pr)
            if i == 1:
                temp_row = ns1[random.randint(0, 34), :]
            if i == 2:
                temp_row = ns2[random.randint(0, 458), :]
            event[row, :] = temp_row
            pass
        # save to file
        np.savetxt("./out/res_faultmaker/scene_result.csv",
                   event, fmt="%i", delimiter=",")
        pass

if __name__ == "__main__":
    a=ScenceMaker()
    a.make()
    pass
