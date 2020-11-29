from lib.GridData import *
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as plot


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
            self.net.load["p_mw"][i] = 1e-6 * data.current_Pload[i]
            self.net.load["q_mvar"][i] = 1e-6 * data.current_Qload[i]
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

        for i in range(0, 36):
            self.net.line["in_service"][i] = data.solution_breaker_state.astype(bool)[
                i]
            pass
        pass

    def cal_reward(self, data: GridData):
        """
        * the function for calculating the reward *
        Reward = (penalty coefficient) * voltagebias + (pirce_loss) * total power loss + (price_blackout) * load loss + (mt_cost) * total P_mt
        """

        reward = self.penalty_voltage * self.voltage_bias + data.price_loss * \
            self.loss_total + data.price_blackout * self.sum_blackout + \
            data.current_price.tolist(
            )[1]*(data.solution_mt_p.tolist()[0]+data.solution_mt_p.tolist()[1])
        return reward

    def render(self, data: GridData):
        """
        * run PF and calculate the reward *
        #TODO Plot 
        """
        print("负载： \n")
        print(self.net.load)
        print("线路： \n")
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
        plot.simple_plot(self.net)
        pass

    def reset(self):
        """
        reset the network
        """
        self.net = pn.case33bw()
        pass

    pass
