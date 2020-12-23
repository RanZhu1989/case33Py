
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
        self.penalty_voltage = 0.5
        self.reward = 0.0
        self.sum_blackout = 0.0
        self.start_time = self.make_time()
        self.env_state_cache = np.array(40)
        pass
    
    def init_action_space(self,path="lib/macro/action_space.csv"):
        """
        read action definition
        
        action number | line_opened#1 | line_opened#2 |...
        
        + return number of actions
        
        """
        self.action_space=np.loadtxt(path,dtype=int,delimiter=",")
        
        
        
        pass

    def init_output(self):
        """
        initialize experience file
        """
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
        self.net.ext_grid["max_p_mw"][0] = 100
        self.net.ext_grid["max_q_mvar"][0] = 100
        self.net.ext_grid["min_q_mvar"][0] = -100
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
        # self.net.bus["max_vm_pu"][3] = 1.0
        # self.net.bus["min_vm_pu"][3] = 1.0
        # self.net.bus["max_vm_pu"][7] = 1.0
        # self.net.bus["min_vm_pu"][7] = 1.0
        # self.net.bus["max_vm_pu"][21] = 1.0
        # self.net.bus["min_vm_pu"][21] = 1.0
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

    def cal_reward(self, data: GridData,silent=True):
        """
        * the function for calculating the reward *
        Reward = (penalty coefficient) * bias of voltage + (pirce_loss) * total power loss 
        + (price_blackout) * load loss + (mt_cost) * total P_mt
        """
        if silent==False:
            print("Load Shed = ", self.sum_blackout)
            print("Voltage Deviation", self.voltage_bias)
            
        reward = (self.penalty_voltage * self.voltage_bias + data.price_loss *
                  self.loss_total * 1e3 + data.price_blackout * self.sum_blackout +
                  data.current_price.tolist(
                  )[1]*sum(data.solution_mt_p)*1e-3)*-1
        return round(reward, 2)

    def check_energized(self):
        """
        Check the island bus
        """

        num_de_energized = len(pt.unsupplied_buses(self.net))
        list_de_energized = list(pt.unsupplied_buses(self.net))
        return num_de_energized, list_de_energized

    def render(self, data: GridData, plot=True, res_print=False, wait_time=None,logger=False):
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
        self.sum_blackout = self.cal_blackout(data)
        # calculate reward
        self.reward = self.cal_reward(data)
        if plot == True:
            self.network_plot(data, mode="color_map", pause=wait_time)
            pass
        pass
        if res_print == True:
            print(self.net.res_bus)
            print("Network Loss = ",self.loss_total*1000)
            print("Loss Rate = ",round(self.loss_total*100 / ( 1e-6*np.real(np.sum(data.current_load)) ),2))
            pass
        if logger==True:
            self.log_data(data)
            pass
    
        pass
    
    def cal_blackout(self,data:GridData,silent=True):
        """
        calculate of outage load
        """
        blackoutnode=np.setdiff1d(np.arange(0,32),np.nonzero(self.net.res_load.sort_index()["p_mw"].tolist())[0])
        if silent==False:
            print("shed_node = ",self.net.res_load.sort_index())
            print("outage_node = ",blackoutnode)
        if len(blackoutnode)==0:
            return 0
        else:
            return np.real(sum(data.current_load[blackoutnode]))*1e-3
    
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
            voltage_map = [((0.00, 0.90), "lime"),((0.90, 0.950), "g"),((0.950, 1.05), "b"),((1.05, 1.1), "m"),((1.1, 1.5), "r")]
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
        if pause == 0:
            plt.show()
        else:
            plt.pause(pause)
            plt.close()
        pass
    def log_data(self,data:GridData):
        """
        log node voltage, network loss and shed load
        """
        
        path_log_voltage="./log/voltage/"+self.start_time + ".csv"
        path_log_loss="./log/loss/"+self.start_time + ".csv"
        path_log_shed="./log/shed/"+self.start_time + ".csv"
        with open(path_log_voltage,"a+") as file:
            for i in range(32):
                if i <31:
                    file.write(str(list(self.net.res_bus.sort_index()["vm_pu"])[i])+",")
                else:
                    file.write(str(list(self.net.res_bus.sort_index()["vm_pu"])[i])+"\n")
                pass
            pass
        with open(path_log_loss,"a+") as file:
            file.write(str(self.loss_total*1000)+",")
            file.write(str(round(self.loss_total*100 / ( 1e-6*np.real(np.sum(data.current_load)) ),2))+"\n")
            pass
        with open(path_log_shed,"a+") as file:
            for i in range(32):
                if i <31:
                    file.write(str(data.solution_loadshed[i])+",")
                else:
                    file.write(str(data.solution_loadshed[i])+"\n")
                pass
            pass
            
            pass
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
                #1. activate power of LOAD in 32 nodes
                #2. activate power output of DGs
                #3. in future the ES will be introduced

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
                    GridData.cache => current_P_load, current_P_PV, current_P_WT, net.line["in_service"]
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
                file.write(str(int(data.pin_cache[i] * 1000)) + ",")
                pass
            for i in range(5):
                file.write(str(data.line_opened_cache[i]) + ",")
                pass
            for i in range(2):
                file.write(str(data.line_fault_cache[i]) + ",")
                pass
            file.write("%i" % (data.current_time-1)+",")
            # save A_t dim=11
            for i in range(3):
                file.write(str(int(data.pmt_cache[i]/1000)) + ",")
                pass
            for i in range(3):
                file.write(str(int(data.qmt_cache[i]/1000)) + ",")
                pass
            for i in range(5):
                file.write(
                    str(np.nonzero(np.ones(37)-data.solution_breaker_state)[0][i]) + ",")
                pass
            # save R_t
            file.write(str(self.reward) + ",")
            # save S_{t+1} dim=39
            temp_pin_next=np.array(self.net.res_bus.sort_index().drop(0, 0)["p_mw"])
            temp_pin_next[2]+=data.solution_mt_p[0]*1e-6
            temp_pin_next[6]+=data.solution_mt_p[1]*1e-6
            temp_pin_next[20]+=data.solution_mt_p[2]*1e-6
            for i in range(32):
                file.write(str(int(temp_pin_next[i]*1000)) + ",")
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

    def env_step(self, t, data: GridData, action: np.ndarray,debug=False,out=False,log=False):
        """
        S1. read current grid data using class GridData;
        S2. data.soultion_xxx covered by input action;
        S3. executes the action using PandapowerTask.rent 
        S4. calculate the reward;
        S5. read the next gird data using class GridData;
        S6. combine the input action with the next uncertain information including load\PV\WT\event 

        return S_t, R_{t+1}, S_{t+1}
        * usage:
        - 1. for online traning mode
        - 2. convert the actions from external solver into MDPC experience file 
        
        # although MOSEK solver is integrated into case33py, it is slow to generate the exp ...
        # we recommend using solver by our case33Julia project, which is more powerful:
            - with DP & MPC based optimizer 
            - more customizable options
            - fast and cache acceleration
            ...
        """
        # set flag_gameover
        flag=False
        # read grid data at t
        data.make_step(step=t)
        # set action
        data.solution_mt_p = 1000*action[0:3]
        data.solution_mt_q = 1000*action[3:6]
        data.solution_breaker_state = np.ones(37, dtype=int)
        data.solution_breaker_state[list(action[6:11])] = 0
        # set parameters
        self.set_parameters(data)
        # executes the action
        # reward has been calculated
        if debug==True:
            self.render(data, plot=True,res_print=True,wait_time=10)
        else:
            self.render(data, plot=False)
        # if number of outage node > 0 or low\hight voltage,  game over
        # or \
        # (len(np.where(np.array(list(self.net.res_bus.sort_index()["vm_pu"]))<0.95)[0])>0) or \
        #     (len(np.where(np.array(list(self.net.res_bus.sort_index()["vm_pu"]))>1.05)[0])>0):
        if  (len(np.nonzero(self.net.res_load.sort_index()["p_mw"].tolist())[0])<32): 
            print(self.net.res_bus)
            print(self.net.res_load)
            flag=True
        # save reward
        save_reward = self.reward
        save_s = self.env_state_cache.copy()
        data.make_step(step=t+1)
        self.set_load(data)
        self.set_pv(data)
        self.set_wt(data)
        pin_next = 1000*np.array(list(self.net.load.sort_index()["p_mw"]))
        # pin_next[2] -= action[0]
        # pin_next[6] -= action[1]
        # pin_next[20] -= action[2]
        # pin opened_line fault_line time
        save_s_next = np.around(np.concatenate(
            (pin_next, action[6:11], data.list_fault_line_number, np.array([t+1, ])))).astype(int)
        self.env_state_cache=save_s_next
        
        
        if out==True:
            with open(self.outpath, "a+") as file:
                # save S_t dim=40
                for i in range(40):
                    file.write(str(save_s[i]) + ",")
                # save A_t dim=11
                for i in range(11):
                    file.write(str(action[i]) + ",")
                # save R_t
                file.write(str(self.reward) + ",")
                # save S_{t+1} dim=40
                for i in range(40):
                    if i <39:
                        file.write(str(save_s_next[i]) + ",")
                    else:
                        file.write(str(save_s_next[i]) + "\n")
                pass
            if log==True:
                self.log_data(data)
            pass
        self.reset()
        return save_s, save_reward, save_s_next, flag

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

    def make_cache(self, data: GridData):
        """
        save the preserving variable such as Breaker_State(S_{t-1}), P_in(S_{t-1}), SOC(S_{t-1}) ...
        """

        # save the current line state using 5 opened breakers
        if len(list(np.nonzero(~data.solution_breaker_state.astype(bool))[0])) != 5:
            print("breaker state error! Exit!")
            sys.exit()
        data.line_opened_cache = np.nonzero(
            ~data.solution_breaker_state.astype(bool))[0]
        data.line_fault_cache = data.list_fault_line_number
        # save the current P_in
        data.pin_cache = np.array(
            self.net.res_bus.sort_index().drop(0, 0)["p_mw"])
        data.pin_cache[2]+= data.solution_mt_p[0]*1e-6
        data.pin_cache[6]+= data.solution_mt_p[1]*1e-6
        data.pin_cache[20]+= data.solution_mt_p[2]*1e-6

        # save mt output
        data.pmt_cache = data.solution_mt_p
        data.qmt_cache = data.solution_mt_q

        pass

    pass

    def init_cache(self, data: GridData,env_mode=False,start=0):
        """
        initialization the cache for the first time
        we assume that the network works in normal condition
        """
        if env_mode==False:
            data.line_opened_cache = np.array([32, 33, 34, 35, 36])
            data.line_fault_cache = np.array([-99, -99])
            data.pin_cache = np.array(
                self.net.load.sort_index()["p_mw"])
            data.pin_cache[2]+= data.solution_mt_p[0]*1e-6
            data.pin_cache[6]+= data.solution_mt_p[1]*1e-6
            data.pin_cache[20]+= data.solution_mt_p[2]*1e-6
            
        else:
            data.make_step(step=start)
            self.set_load(data)
            self.set_pv(data)
            self.set_wt(data)
            pin = 1000*np.array(list(self.net.load.sort_index()["p_mw"]))
            self.env_state_cache=np.around(np.concatenate(
            (pin, np.array([32, 33, 34, 35, 36]), np.array([-99, -99]), np.array([0, ])))).astype(int)
            pass

        pass

    def action_mapping(self,action,mode="bounded"):
        """
        transform action vector into real action for RL  
        
        - in bounded mode:
            reference = [P]
            
        """
        out=np.zeros(11)
        if mode=="bounded":
            for i in range(3):
                out[i]=(abs(action[i])+1)*1000
                pass
            for i in range(3,6):
                out[i]=(abs(action[i])+1)*500
                pass
            
            out[6:12]=self.action_space[int(action[6]),:]
            
            pass
            
            
            
        return out.astype("int")
    
    
if __name__ == "__main__":

    pass
