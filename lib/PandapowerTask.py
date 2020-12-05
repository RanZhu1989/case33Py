
from os import write
import time
from lib.GridData import *
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as pplot
import pandapower.topology as pt
import matplotlib.pyplot as plt
try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]


class PandapowerTask():
    """
    * An nonlinear power flow interactive environment powered by pandapower 2.4 *
    PandapowerTask will be called after the optimization is completed

    """

    def __init__(self):
        """
        create IEEE33bw standard model
        """
        self.net = pn.case33bw()
        # sort the index
        self.net.load = self.net.load.sort_index()
        self.net.line = self.net.line.sort_index()
        self.loss_total = 0.0
        self.voltage_bias = 0.0
        self.blackout_power = 0.0
        self.penalty_voltage = 1.0
        self.reward = 0.0
        self.sum_blackout = 0.0
        self.start_time=self.make_time()
        pass

    def set_parameters(self, data: GridData):
        """
        set parameters of the network
        """
        self.set_load(data)
        self.set_mt(data)
        self.set_pv(data)
        self.set_wt(data)
        self.set_breaker(data)
        pass

    def set_load(self, data: GridData):
        """
        set load in current step
        """
        for i in range(0, 32):
            self.net.load["p_mw"][i] = 1e-6 * data.current_Pload[i] *data.solution_loadshed[i]
            self.net.load["q_mvar"][i] = 1e-6 * data.current_Qload[i]*data.solution_loadshed[i]
            pass
        pass

    def set_mt(self, data: GridData):
        """
        set MTs as PQ nodes 
        """

        self.net.load["p_mw"][3] -= 1e-6 * data.solution_mt_p[0]
        self.net.load["q_mvar"][3] -= 1e-6 * data.solution_mt_q[0]
        self.net.load["p_mw"][7] -= 1e-6 * data.solution_mt_p[1]
        self.net.load["q_mvar"][7] -= 1e-6 * data.solution_mt_q[1]
        self.net.load["p_mw"][21] -= 1e-6 * data.solution_mt_p[2]
        self.net.load["q_mvar"][21] -= 1e-6 * data.solution_mt_q[2]
        pass

    def set_pv(self, data: GridData):
        """
        set PVs as PQ nodes
        """

        self.net.load["p_mw"][7] -= 1e-6 * np.real(data.current_pv[0])
        self.net.load["q_mvar"][7] -= 1e-6 * np.imag(data.current_pv[0])
        self.net.load["p_mw"][13] -= 1e-6 * np.real(data.current_pv[1])
        self.net.load["q_mvar"][13] -= 1e-6 * np.imag(data.current_pv[1])
        pass

    def set_wt(self, data: GridData):
        """
        set WTs as PQ nodes
        """

        self.net.load["p_mw"][27] -= 1e-6 * np.real(data.current_wt[0])
        self.net.load["q_mvar"][27] -= 1e-6 * np.imag(data.current_wt[0])
        self.net.load["p_mw"][29] -= 1e-6 * np.real(data.current_wt[1])
        self.net.load["q_mvar"][29] -= 1e-6 * np.imag(data.current_wt[1])
        pass

    def set_breaker(self, data: GridData):
        """
        set breaker state
        """

        for i in range(37):
            self.net.line["in_service"][i] = data.solution_breaker_state.astype(bool)[
                i]
            pass
        pass

    def cal_reward(self, data: GridData):
        """
        * the function for calculating the reward *
        Reward = (penalty coefficient) * bias of voltage + (pirce_loss) * total power loss 
        + (price_blackout) * load loss + (mt_cost) * total P_mt
        """
        print("负荷损失为", self.sum_blackout)
        reward = (self.penalty_voltage * self.voltage_bias + data.price_loss *
                  self.loss_total * 1e3 + data.price_blackout * self.sum_blackout*1e3 +
                  data.current_price.tolist(
                  )[1]*(data.solution_mt_p.tolist()[0]+data.solution_mt_p.tolist()[1])*1e-3)*-1
        return reward

    def check_energized(self):
        """
        Check the island bus
        """

        num_de_energized = len(pt.unsupplied_buses(self.net))
        list_de_energized = list(pt.unsupplied_buses(self.net))
        return num_de_energized, list_de_energized

    def render(self, data: GridData):
        """
        * run PF and calculate the reward *
        #TODO Plot 
        """
        print("load： \n")
        print(self.net.load)
        print("line： \n")
        print(self.net.line)

        # run PF
        pp.runpp(self.net)
        # calculate total power loss
        self.loss_total = sum(self.net.res_line["pl_mw"])
        # calculate voltage bias
        v_sum = 0
        for v in self.net.res_bus["vm_pu"]:
            if not np.isnan(v):
                v_sum += max(0, v - max(self.net.res_bus["vm_pu"])) + \
                    max(0, min(self.net.res_bus["vm_pu"]) - v)
                pass
            pass
        self.voltage_bias = v_sum
        # calculate blackout loss
        self.sum_blackout = sum(
            self.net.res_bus["p_mw"][1:, ])-sum(self.net.load["p_mw"][0:, ])
        # calculate reward
        self.reward = self.cal_reward(data)
        print(self.reward)
        print(self.net.res_bus)
        print(self.net.res_line)
        print(self.net.res_load)
        self.network_plot(data, mode="color_map")
        pass

    def reset(self):
        """
        reset the network
        """
        self.net = pn.case33bw()
        # sort the index
        self.net.load = self.net.load.sort_index()
        self.net.line = self.net.line.sort_index()
        pass

    def network_plot(self, data: GridData, mode="topological_graph",pause=5):
        """
        plot the network using matplotlib

        * mode = topological_graph or color_map
        """

        if mode == "topological_graph":
            self.net.bus_geodata.drop(self.net.bus_geodata.index, inplace=True)
            self.net.line_geodata.drop(
                self.net.line_geodata.index, inplace=True)
            pplot.create_generic_coordinates(self.net)
            pplot.fuse_geodata(self.net)
            buses = self.net.bus.index.tolist()
            coords = zip(
                self.net.bus_geodata.x.loc[buses].values, self.net.bus_geodata.y.loc[buses].values)
            bus_layer = pplot.create_bus_collection(
                self.net, self.net.bus.index, size=.05, color="black", zorder=1)
            sub_layer = pplot.create_bus_collection(
                self.net, self.net.ext_grid.bus.values, patch_type="rect", size=.2, color="yellow", zorder=2)
            busid_layer = pplot.create_annotation_collection(
                size=0.2, texts=np.char.mod('%d', buses), coords=coords, zorder=3, color="blue")
            line_layer = pplot.create_line_collection(
                self.net, self.net.line.index, color="grey", linestyles="dashed", linewidths=0.2, use_bus_geodata=True, zorder=4)
            lines_ergized = self.net.line[self.net.line.in_service == True].index
            line_ergized_layer = pplot.create_line_collection(
                self.net, lines_ergized, color="red", zorder=5)
            pplot.draw_collections(
                [line_layer, bus_layer, sub_layer, busid_layer, line_ergized_layer], figsize=(8, 6))
            pass

        if mode == "color_map":
            self.net.bus_geodata.drop(self.net.bus_geodata.index, inplace=True)
            self.net.line_geodata.drop(
                self.net.line_geodata.index, inplace=True)
            voltage_map = [((0.975, 0.985), "blue"),
                           ((0.985, 1.0), "green"), ((1.0, 1.03), "red")]
            cmap, norm = pplot.cmap_discrete(voltage_map)
            pplot.create_generic_coordinates(self.net)
            pplot.fuse_geodata(self.net)
            buses = self.net.bus.index.tolist()
            coords = zip(
                self.net.bus_geodata.x.loc[buses].values, self.net.bus_geodata.y.loc[buses].values)
            bus_layer = pplot.create_bus_collection(
                self.net, self.net.bus.index, size=.05, cmap=cmap, norm=norm, color="black", zorder=1)
            sub_layer = pplot.create_bus_collection(
                self.net, self.net.ext_grid.bus.values, patch_type="rect", size=.2, color="yellow", zorder=2)
            busid_layer = pplot.create_annotation_collection(
                size=0.2, texts=np.char.mod('%d', buses), coords=coords, zorder=3, color="blue")
            line_layer = pplot.create_line_collection(
                self.net, self.net.line.index, color="grey", linestyles="dashed", linewidths=0.2, use_bus_geodata=True, zorder=4)
            lines_ergized = self.net.line[self.net.line.in_service == True].index
            line_ergized_layer = pplot.create_line_collection(
                self.net, lines_ergized, color="red", zorder=5)
            pplot.draw_collections(
                [line_layer, bus_layer, sub_layer, busid_layer, line_ergized_layer], figsize=(8, 6))
            pass

        plt.ion()
        plt.plot()
        # put the fault lines list in the figure
        plt.annotate("fault lines: %s" % data.list_fault_line, (-2.3, -2.8))
        plt.pause(pause)
        plt.close()
        pass
        
    def out2file(self):
        """
        save MDP chain (S_t,A_t,R_t,S_t+1) to csv file
        
        ** S = { P_i, (Toplogy, Fault_lines) }  dim = 39
            + P_i is the aggregation of:
                1. activate power of load in 32 nodes
                2. activate power output of PV and WT
                3. in future the ES will be introduced
            + Toplogy is a set of 7 breakers 
            # in the worst case, 7 breakers could be opened (5 for normal constraint + 2 for fault) 
             (at present, we focus on N-1 and N-2 problems in IEEE33BW case) 
                Because |line| = 37 , |node| =33 , the ST constraint makes always 32 breakers closed 
                to make full load restroation.
            + Fault_lines is a list of fault breakers (at most 2)
                in case33bw, |line| = 37 
                the number of fault senses aviliable is almost to 500
                e.g.  
                      only line#1 fail => ( 1, 0 ) 
                      line#2 and line#3 fail =>( 2, 3)    
                      normal operation => ( 0, 0 )
            !!S_t comes from the solution of the previous step, while S_t+1 is taken from the current solution:
                1. to form S_t:
                GridData.cash => current_P_load, current_P_PV, current_P_WT, net.line["in_service"]
                GridData.current_event => list_fault_lines( dim=2 )
                2. to form S_t+1:
                GridData.current_xxx => current_P_load, current_P_PV, current_P_WT
                PandapowerTask.net.line["in_service"]
                
                                 
               
                    
        ** A = { Pmt_i, Qmt_i, Alpha_i }
            + Pmt_i and Qmt_i are the three MTs' output
            + Alpha_i is a set of 5 breakers
             (at present, we focus on N-1 and N-2 problems in IEEE33BW case) 
                Because |line| = 37 , |node| =33 , the ST constraint makes always 32 breakers closed 
                to make full load restroation.
             
        """
        path="./out/res_MDPC/"+self.start_time+ "prim_MDPC.csv"
        with open(path,"a+") as file:
            for i in range(33):     
                file.write(str(self.net.res_bus.sort_index()["vm_pu"][i])+",")
                pass
            for i in range(33): 
                file.write(str(self.net.res_bus.sort_index()["p_mw"][i])+",")
                pass
            for i in range(33): 
                file.write(str(self.net.res_bus.sort_index()["q_mvar"][i])+",")
                pass
            for i in range(37): 
                if i<36:
                    file.write(str(self.net.line.sort_index()["in_service"][i])+",")
                else:file.write(str(self.net.line.sort_index()["in_service"][i]))    
                pass
            file.write("\n")
            pass
        pass

    def make_time(self):
        """
        return a list about current time
        """
        current_time=time.localtime(time.time())
        y=current_time[0]
        mon=current_time[1]
        d=current_time[2]
        h=current_time[3]
        m=current_time[4]
        res=str(y)+str(mon)+str(d)+"time"+str(h)+"_"+str(m)
        return  res

    pass


if __name__ == "__main__":

    pass
