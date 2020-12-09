
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
        self.net.bus = self.net.bus.sort_index()
        self.loss_total = 0.0
        self.voltage_bias = 0.0
        self.blackout_power = 0.0
        self.penalty_voltage = 1.0
        self.reward = 0.0
        self.sum_blackout = 0.0
        self.start_time = self.make_time()
        self.outpath = "./out/res_MDPC/"+self.start_time + "prim_MDPC.csv"
        # init the header of MDPC.csv
        with open(self.outpath, "a+") as file:
            for i in range(1, 33):
                file.write("S(t)_Pin_%i" % i + ",")
                pass
            for i in range(1, 6):
                file.write("S(t)_opened_line_%i" % i + ",")
                pass
            for i in range(1, 3):
                file.write("S(t)_failed_line_%i" % i + ",")
                pass
            file.write("current_time"+",")
            for i in range(1, 4):
                file.write("A(t)_Pmt_%i" % i + ",")
                pass
            for i in range(1, 4):
                file.write("A(t)_Qmt_%i" % i + ",")
                pass
            for i in range(1, 6):
                file.write("A(t)_open_line_%i" % i + ",")
                pass
            file.write("R(t+1)"+",")
            for i in range(1, 33):
                file.write("S(t+1)_Pin_%i" % i + ",")
                pass
            for i in range(1, 6):
                file.write("S(t+1)_opened_line_%i" % i + ",")
                pass
            for i in range(1, 3):
                file.write("S(t+1)_failed_line_%i" % i + ",")    
                pass
            file.write("next_time")
            file.write("\n")
            pass
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
            self.net.load["p_mw"][i] = 1e-6 * \
                data.current_Pload[i] * data.solution_loadshed[i]
            self.net.load["q_mvar"][i] = 1e-6 * \
                data.current_Qload[i]*data.solution_loadshed[i]
            pass
        pass

    def set_mt(self, data: GridData):
        """
        MTs have fixed output and make the node with fixed voltage at 1.0 pu.
        """

        self.net.load["p_mw"][2] -= 1e-6 * data.solution_mt_p[0]
        self.net.load["q_mvar"][2] -= 1e-6 * data.solution_mt_q[0]
        self.net.load["p_mw"][6] -= 1e-6 * data.solution_mt_p[1]
        self.net.load["q_mvar"][6] -= 1e-6 * data.solution_mt_q[1]
        self.net.load["p_mw"][20] -= 1e-6 * data.solution_mt_p[2]
        self.net.load["q_mvar"][20] -= 1e-6 * data.solution_mt_q[2]
        self.net.bus["max_vm_pu"][3]=1.0
        self.net.bus["min_vm_pu"][3]=1.0
        self.net.bus["max_vm_pu"][7]=1.0
        self.net.bus["min_vm_pu"][7]=1.0
        self.net.bus["max_vm_pu"][21]=1.0
        self.net.bus["min_vm_pu"][21]=1.0
        pass

    def set_pv(self, data: GridData):
        """
        set PVs as PQ nodes
        """

        self.net.load["p_mw"][6] -= 1e-6 * np.real(data.current_pv[0])
        self.net.load["q_mvar"][6] -= 1e-6 * np.imag(data.current_pv[0])
        self.net.load["p_mw"][12] -= 1e-6 * np.real(data.current_pv[1])
        self.net.load["q_mvar"][12] -= 1e-6 * np.imag(data.current_pv[1])
        pass

    def set_wt(self, data: GridData):
        """
        set WTs as PQ nodes
        """

        self.net.load["p_mw"][26] -= 1e-6 * np.real(data.current_wt[0])
        self.net.load["q_mvar"][26] -= 1e-6 * np.imag(data.current_wt[0])
        self.net.load["p_mw"][28] -= 1e-6 * np.real(data.current_wt[1])
        self.net.load["q_mvar"][28] -= 1e-6 * np.imag(data.current_wt[1])
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
        print("电压偏差指标为",self.voltage_bias)
        reward = (self.penalty_voltage * self.voltage_bias + data.price_loss *
                  self.loss_total * 1e3 + data.price_blackout * self.sum_blackout*1e3 +
                  data.current_price.tolist(
                  )[1]*(data.solution_mt_p.tolist()[0]+data.solution_mt_p.tolist()[1])*1e-3)*-1
        return round(reward, 2)

    def check_energized(self):
        """
        Check the island bus
        """

        num_de_energized = len(pt.unsupplied_buses(self.net))
        list_de_energized = list(pt.unsupplied_buses(self.net))
        return num_de_energized, list_de_energized

    def render(self, data: GridData, plot=True):
        """
        * run PF and calculate the reward *
        """
        # run PF
        pp.runpp(self.net)
        # calculate total power loss
        self.loss_total = sum(self.net.res_line["pl_mw"])
        # calculate voltage bias
        v_sum = 0
        for v in self.net.res_bus["vm_pu"]:
            if not np.isnan(v):
                v_sum += max(0, v - 1.05) + max(0, 0.95 - v)
                pass
            pass
        self.voltage_bias = v_sum
        # calculate blackout loss
        self.sum_blackout = round(10*sum(self.net.load["p_mw"][0:, ])-sum(
            self.net.res_bus["p_mw"][1:, ])*1000, 2)
        # calculate reward
        self.reward = self.cal_reward(data)
        if plot == True:
            self.network_plot(data, mode="color_map")
            pass
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

    def network_plot(self, data: GridData, mode="topological_graph", pause=5):
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

    def exp_out2file(self, data: GridData):
        """
        save MDP chain (S_t,A_t,R_t,S_t+1) to csv file

        **   S_t  ** **  A_t  **   ** R_t **  ** S_{t+1} **
        |- [0,39] -| |- [40,50] -| |- [51] -| |- [52,91] -|
        -------------------------------------------------->

        !! the unit of the power set to kW !!

        ** S = { P_i Toplogy, Fault_lines) }  dim = 32 + 5 + 2 + 1 = 40
            + P_i is the aggregation of:
                #1. activate power of load in 32 nodes
                #2. activate power output of PV and WT
                #3. in future the ES will be introduced
                !! NOT include MT !!
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
                # 1. to form S_t:
                    GridData.cash => current_P_load, current_P_PV, current_P_WT, net.line["in_service"]
                    GridData.current_event => list_fault_lines( dim=2 )
                # 2. to form S_t+1:
                    GridData.current_xxx => current_P_load, current_P_PV, current_P_WT
                    PandapowerTask.net.line["in_service"]
            # The last term of S_t or S_{t+1} is time - the unique state 

        ** A = { Pmt_i, Qmt_i, Alpha_i } dim = 3 + 3 + 5 = 11 
            + Pmt_i and Qmt_i are the three MTs' output
            + Alpha_i is a set of 5 breakers
             (at present, we focus on N-1 and N-2 problems in IEEE33BW case) 
                Because |line| = 37 , |node| =33 , the ST constraint makes always 32 breakers closed 
                to make full load restroation.

        """

        with open(self.outpath, "a+") as file:
            # save S_t dim=39
            for i in range(32):
                file.write(str(int(data.pin_cash[i] * 1000)) + ",")
                pass
            for i in range(5):
                file.write(str(data.line_opened_cash[i]) + ",")
                pass
            for i in range(2):
                file.write(str(data.line_fault_cash[i]) + ",")
                pass
            file.write("%i" % (data.current_time-1)+",")
            # save A_t dim=11
            for i in range(3):
                file.write(str(int(data.pmt_cash[i]/1000)) + ",")
                pass
            for i in range(3):
                file.write(str(int(data.qmt_cash[i]/1000)) + ",")
                pass
            for i in range(5):
                file.write(
                    str(np.nonzero(np.ones(37)-data.solution_breaker_state)[0][i]) + ",")
                pass
            # save R_t
            file.write(str(self.reward) + ",")
            # save S_{t+1} dim=39
            for i in range(32):
                file.write(str(
                    int(np.array(self.net.res_bus.sort_index().drop(0, 0)["p_mw"])[i]*1000)) + ",")
                pass
            for i in range(5):
                file.write(
                    str(np.nonzero(np.ones(37)-data.solution_breaker_state)[0][i]) + ",")
                pass
            for i in range(2):
                file.write(
                    str(np.array(data.list_fault_line_number)[i]) + ",")
            file.write("%i" % data.current_time)
            file.write("\n")
            pass
        pass

    def env_step(self, t,data: GridData,action:np.ndarray):
        """
        S1. read current grid data using class GridData;
        S2. data.soultion_xxx covered by input action;
        S3. executes the action using PandapowerTask.rent 
        S4. calculate the reward;
        S5. read the next gird data using class GridData;
        S6. combine the input action with the next uncertain information including load\PV\WT\event 

        return S_t, R_{t+1}, S_{t+1}
        * used for online traning mode
        """
        # read grid data at t
        data.make_step(step=t)
        # set action
        data.solution_mt_p=1000*action[0:3]
        data.solution_mt_q=1000*action[3:6]
        data.solution_breaker_state=np.ones(37,dtype=int)
        data.solution_breaker_state[list(action[6:11])]=0
        # set parameters
        self.set_parameters(data)
        # executes the action
        # reward has been calculated
        self.render(data,plot=False)
        # save reward
        save_reward=self.reward
        fault_line=np.zeros(2,dtype=int)
        fault_line_next=np.zeros(2,dtype=int)
        if len(data.list_fault_line_number) == 0:
            fault_line = np.pad(
                data.list_fault_line_number, (0, 2), mode="constant", constant_values=(-99))
        else:
            if len(data.list_fault_line_number) == 1:
                fault_line = np.pad(
                    data.list_fault_line_number, (0, 1), mode="constant", constant_values=(-99))
            else:
                fault_line = data.list_fault_line_number
                pass
            pass
        pin=1000*np.array(list(self.net.load.sort_index()["p_mw"]))
        # drop the mt output
        pin[2]+=action[0]
        pin[6]+=action[1]
        pin[20]+=action[2]
        # pin opened_line fault_line time  
        save_s=np.around(np.concatenate((pin,action[6:11],fault_line,np.array([t,])))).astype(int)
        data.make_step(step=t+1)
        self.set_load(data)
        self.set_mt(data)
        self.set_pv(data)
        self.set_wt(data)
        pin_next=1000*np.array(list(self.net.load.sort_index()["p_mw"]))
        if len(data.list_fault_line_number) == 0:
            fault_line_next = np.pad(
                data.list_fault_line_number, (0, 2), mode="constant", constant_values=(-99))
        else:
            if len(data.list_fault_line_number) == 1:
                fault_line_next = np.pad(
                    data.list_fault_line_number, (0, 1), mode="constant", constant_values=(-99))
            else:
                fault_line_next = data.list_fault_line_number
                pass
            pass
        # pin opened_line fault_line time
        save_s_next=np.around(np.concatenate((pin_next,action[6:11],fault_line_next,np.array([t+1,])))).astype(int)
        
        return save_s,save_reward,save_s_next


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
        res = str(y)+str(mon)+str(d)+"time"+str(h)+"_"+str(m)
        return res

    pass

    def make_cash(self, data: GridData):
        """
        save the preserving variable such as Breaker_State(S_{t-1}), P_in(S_{t-1}), SOC(S_{t-1}) ...
        """

        # save the current line state using 5 opened breakers
        if len(list(np.nonzero(~data.solution_breaker_state.astype(bool))[0])) != 5:
            sys.exit()
        data.line_opened_cash = np.nonzero(
            ~data.solution_breaker_state.astype(bool))[0]

        # save the current fault line using pad method
        # to distinguish between noramal state and fault state, we use -99 as a place holder

        if len(data.list_fault_line_number) == 0:
            data.line_fault_cash = np.pad(
                data.list_fault_line_number, (0, 2), mode="constant", constant_values=(-99))
        else:
            if len(data.list_fault_line_number) == 1:
                data.line_fault_cash = np.pad(
                    data.list_fault_line_number, (0, 1), mode="constant", constant_values=(-99))
            else:
                data.line_fault_cash = data.list_fault_line_number
                pass
            pass

        # save the current P_in
        data.pin_cash = np.array(
            self.net.res_bus.sort_index().drop(0, 0)["p_mw"])

        # save mt output
        data.pmt_cash = data.solution_mt_p
        data.qmt_cash = data.solution_mt_q

        pass

    pass


if __name__ == "__main__":

    pass
