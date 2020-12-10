from mosek.fusion import *
from lib.GridData import *


class MosekOPF:
    """
    class of the optimization problem modeled with fusion in normal and slight fault
    in this class, the power flow is modeled as Dist-Flow, which will be a SOCP optimization problem
    """

    def __init__(self, current_data: GridData, v_range=((0.95 * 12.66e3) ** 2, (1.05 * 12.66e3) ** 2),
                 f_range=(49.75, 50.25), i_max=100.0, soc_range=(0.0, 1.0)):
        """
        initialization a fusion model with parameters:
        v_range
        f_range
        i_max
        soc_range
        """
        print("Current num_lines_alive = ", current_data.num_lines)

        # build a model
        self.model = Model("case33")
        # sqr of node voltage
        self.v_sqr = self.model.variable(
            "v_sqr", 33, Domain.inRange(v_range[0], v_range[1]))
        # sqr of line current
        self.i_sqr = self.model.variable(
            "i_sqr", current_data.num_lines, Domain.inRange(0.0, i_max ** 2))
        # frequent
        # self.frequency = self.model.variable(
        #     "frequency", Domain.inRange(f_range[0], f_range[1]))
        # SOC
        # self.soc=self.M.variable("soc",2,Domain.inRange(soc_range[0],soc_range[1]))
        # P_mt
        self.p_mt = self.model.variable("p_mt", 3)
        # Q_mt
        self.q_mt = self.model.variable("q_mt", 3)
        # load-shedding state , the first element fixed to 0
        self.load_shed = self.model.variable("load_shed", 32, Domain.binary())
        # breaker state
        self.alpha = self.model.variable(
            "alpha", current_data.num_lines, Domain.binary())
        # auxiliary variable for ST constraints
        self.beta = self.model.variable(
            "beta", current_data.num_lines * 2, Domain.binary())
        # fictitious flow for SCF
        self.f_flow = self.model.variable("f_flow", current_data.num_lines)

        # ES charge flag and discharge flag
        # self.charge_flag=self.M.variable("charge_flag",2,Domain.binary())
        # self.discharge_flag=self.M.variable("discharge_flag",2,Domain.binary())

        # forward activate & reactivate power are free
        self.p_forward = self.model.variable(
            "p_forward", current_data.num_lines)
        self.q_forward = self.model.variable(
            "q_forward", current_data.num_lines)
        # inject activate & reactivate power
        self.p_in = self.model.variable("p_in", 33)
        self.q_in = self.model.variable("q_in", 33)

        # big M
        self.bigM = 1e7

        pass

    def make_constraints(self, current_data: GridData):

        # Do some preparation
        # set mask matrix for SOMETHING_ij ,SOMETHING_jk , mask matrix of r\x\z^2
        self.mask_matrix_ij, self.mask_matrix_r, self.mask_matrix_x, self.mask_matrix_sqrz, self.mask_matrix_i = current_data.make_matrix(
            mode="ij_forward")
        self.mask_matrix_jk = current_data.make_matrix(mode="jk_forward")

        # set the line mask matrix "root-free" in ij\jk mode
        self.mask_matrix_ij_rootfree = np.delete(self.mask_matrix_ij, 0, 0)
        self.mask_matrix_jk_rootfree = np.delete(self.mask_matrix_jk, 0, 0)

        # set mask matrix for topological constraints in pairs mode
        self.mask_matrix_pairs = current_data.make_matrix(mode="pairs")

        # set the mask matrix "root-free" in pairs mode
        self.mask_matrix_pairs_rootfree = np.delete(
            self.mask_matrix_pairs, 0, 0)
        '''
        set the injection power
        default set is:
        # substation = { node[1] } without any load
        # MT = { node[4]; node[8]; node[22] }
        # WT = { node[28]; node[33] }
        # PV = { node[8] ; node[14] }
        # power_in = power_{sub\MT\WT\PV} - power_load
        '''
        # for MT-installed nodes

        st_mt_pin = self.model.constraint(Expr.sub(
            Expr.add(Expr.sub(self.p_mt, Expr.mulElm(self.load_shed.pick([3, 7, 21]), current_data.list_Pload_MT.tolist(
            ))), current_data.list_Ppv_MT.tolist()), self.p_in.pick([3, 7, 21])), Domain.equalsTo(0.0))
        st_mt_qin = self.model.constraint(Expr.sub(
            Expr.add(Expr.sub(self.q_mt, Expr.mulElm(self.load_shed.pick([3, 7, 21]), current_data.list_Qload_MT.tolist(
            ))), current_data.list_Qpv_MT.tolist()), self.q_in.pick([3, 7, 21])), Domain.equalsTo(0.0))

        # for MT-free nodes
        # P_pv + P_wt - shed * P_load = P_in
        st_pin = self.model.constraint(Expr.sub(Expr.sub(Expr.add(
            Expr.mulElm(self.load_shed.pick(
                current_data.list_in_loadshed.tolist()), current_data.list_Pload.tolist()),
            self.p_in.pick(
                current_data.list_in.tolist())), current_data.list_Ppv.tolist()), current_data.list_Pwt.tolist()),
            Domain.equalsTo(0.0))
        st_qin = self.model.constraint(Expr.sub(Expr.sub(Expr.add(
            Expr.mulElm(self.load_shed.pick(
                current_data.list_in_loadshed.tolist()), current_data.list_Qload.tolist()),
            self.q_in.pick(
                current_data.list_in.tolist())), current_data.list_Qpv.tolist()), current_data.list_Qwt.tolist()),
            Domain.equalsTo(0.0))

        # Dist-Flow power constraints

        st_df_p = self.model.constraint(Expr.sub(
            Expr.sub(Expr.add(self.p_in, Expr.mul(self.mask_matrix_ij.tolist(), self.p_forward)),
                     Expr.mul(self.mask_matrix_r.tolist(), self.i_sqr)),
            Expr.mul(self.mask_matrix_jk.tolist(), self.p_forward)), Domain.equalsTo(0.0))
        st_df_q = self.model.constraint(Expr.sub(
            Expr.sub(Expr.add(self.q_in, Expr.mul(self.mask_matrix_ij.tolist(), self.q_forward)),
                     Expr.mul(self.mask_matrix_x.tolist(), self.i_sqr)),
            Expr.mul(self.mask_matrix_jk.tolist(), self.q_forward)), Domain.equalsTo(0.0))

        # Dist-Flow node voltage constraints
        # test=self.model.constraint(Expr.mul(50,Expr.sub(1,self.alpha.index(37))),Domain.equalsTo(0.0))

        for idx in range(0, current_data.num_lines):
            ij, r, x, z_sqr = current_data.lookup(idx)
            self.model.constraint(Expr.add(Expr.sub(Expr.add(Expr.sub(self.v_sqr.index(ij[0]-1), self.v_sqr.index(ij[1]-1)), Expr.mul(self.bigM, Expr.sub(
                1, self.alpha.index(idx)))), Expr.mul(2, Expr.add(Expr.mul(r, self.p_forward.index(idx)), Expr.mul(x, self.q_forward.index(idx))))), Expr.mul(z_sqr, self.i_sqr.index(idx))), Domain.greaterThan(0.0))
            self.model.constraint(Expr.add(Expr.sub(Expr.sub(Expr.sub(self.v_sqr.index(ij[0]-1), self.v_sqr.index(ij[1]-1)), Expr.mul(self.bigM, Expr.sub(
                1, self.alpha.index(idx)))), Expr.mul(2, Expr.add(Expr.mul(r, self.p_forward.index(idx)), Expr.mul(x, self.q_forward.index(idx))))), Expr.mul(z_sqr, self.i_sqr.index(idx))), Domain.lessThan(0.0))
            pass

        # R-SOC constraints
        for j in range(0, 33):
            for jk in np.nonzero(current_data.seek_neighbor(j + 1, mode="jk_forward"))[0].tolist():
                self.model.constraint(Expr.mulElm([0.5, 1, 1, 1], Var.vstack(
                    Var.vstack(Var.vstack(self.v_sqr.index(j),
                                          self.i_sqr.index(jk)), self.p_forward.index(jk)),
                    self.q_forward.index(jk))), Domain.inRotatedQCone(4))
                pass
            pass

        # ST
        # beta_ij + beta_ji = alpha_ij
        for idx in range(0, current_data.num_lines):
            self.model.constraint(Expr.sub(Expr.add(self.beta.index(idx), self.beta.index(
                idx+current_data.num_lines)), self.alpha.index(idx)), Domain.equalsTo(0))
            pass
        # sum( beta_ij ) = 1, for each i!=root
        st_st2 = self.model.constraint(Expr.mul(
            self.mask_matrix_pairs_rootfree.tolist(), self.beta), Domain.equalsTo(1))
        # beta_ij = 0, i==root
        st_st3 = self.model.constraint(self.beta.index(0), Domain.equalsTo(0))

        # SCF
        st_scf1 = self.model.constraint(Expr.sub(Expr.mul(self.mask_matrix_jk_rootfree, self.f_flow), Expr.mul(
            self.mask_matrix_ij_rootfree, self.f_flow)), Domain.equalsTo(-1.0))
        st_scf2_1 = self.model.constraint(Expr.sub(self.f_flow, Expr.mulElm(
            self.alpha, list(self.bigM * np.ones(current_data.num_lines)))), Domain.lessThan(0.0))
        st_scf2_2 = self.model.constraint(Expr.add(self.f_flow, Expr.mulElm(
            self.alpha, list(self.bigM * np.ones(current_data.num_lines)))), Domain.greaterThan(0.0))
        st_scf3 = self.model.constraint(
            Expr.sum(self.alpha), Domain.equalsTo(32))

        # Q-V droop control for PV & WT

        # output of MTs
        st_pmt = self.model.constraint(self.p_mt, Domain.inRange(
            [0, 0, 0], [1e6, 1e6, 1e6]))
        st_qmt = self.model.constraint(self.q_mt, Domain.inRange(
            [0, 0, 0], [0.5e6, 0.5e6, 0.5e6]))

        # substaion
        # v_sub = 1.0 pu
        st_sub1 = self.model.constraint(
            self.v_sqr.index(0), Domain.equalsTo(12.66e3 ** 2))

    def make_objective(self, current_data: GridData):
        """
        set the optimization objective function

        obj = Minimize   Price_G * P_sub + sum( Cost_MT * P_mt) + Cost_loss * P_loss + Cost_blackout * P_blackout

        where:
        P_loss = sum( i_sqr * r_ij )
        P_blackout = sum( (1-load_shed) * P_load)

        """

        obj_function = Expr.add(Expr.add(Expr.add(Expr.mul(current_data.current_price.tolist()[0], self.p_in.index(0)),
                                                  Expr.sum(Expr.mulElm(
                                                      (current_data.current_price.tolist()[
                                                       1] * np.ones(3)).tolist(),
                                                      self.p_mt))), Expr.mul(
            current_data.price_loss, Expr.sum(Expr.mulElm(self.i_sqr, current_data.list_r.tolist())))),
            Expr.mul(current_data.price_blackout, Expr.sum(
                Expr.mulElm(Expr.sub(np.ones(32).tolist(), self.load_shed),
                            current_data.current_Pload.tolist()))))
        self.model.objective("obj", ObjectiveSense.Minimize, obj_function)
        pass

    def solve(self, step, current_data: GridData, log=False, debug=False):
        """
        run the solver and get the solution

        - Optional - 
        write the log in OPF file, which is named by "step"

        Once the solution is OPTIMAL, the solution will be saved within GridData Class


        """
        self.model.setSolverParam("intpntCoTolRelGap", 5e-1)
        self.model.solve()

        # write the log file
        if log == True:
            self.model.writeTask("./log/step_%d_test.opf" % (step+1))
            pass

        if debug == True:
            print(str(self.model.getProblemStatus()))
            print(str(self.model.getPrimalSolutionStatus()))
            print(list(self.alpha.level()))
            print(self.load_shed.level())
            print([sqrt(i)/12.66e3 for i in list(self.v_sqr.level())])
            print(self.p_mt.level())
            print(self.q_mt.level())
            pass

        if str(self.model.getPrimalSolutionStatus()) == "SolutionStatus.Optimal":
            current_data.solution_mt_p = list(self.p_mt.level())
            current_data.solution_mt_p = np.around(
                current_data.solution_mt_p, 2)
            current_data.solution_mt_q = list(self.q_mt.level())
            current_data.solution_mt_q = np.around(
                current_data.solution_mt_q, 2)
            current_data.solution_loadshed = list(self.load_shed.level())
            current_data.map_lines(self.alpha.level())
        else:
            print("No soultion! Exit")
            sys.exit()

        if debug == True:
            print(current_data.solution_breaker_state)
            print(np.nonzero(current_data.solution_breaker_state))
            pass

        pass

    pass


'''
!!! DONT USE !!!
MosekDNR Class is still in work progress
'''
class MosekDNR(MosekOPF):
    '''
    inherited from class MosekOPF
    ** network try to pick up loads as much as possible adaptive.
       system could be divided into several islands
    '''

    def __init__(self, current_data: GridData, v_range=((0.95 * 12.66e3) ** 2, (1.05 * 12.66e3) ** 2),
                 f_range=(49.75, 50.25), i_max=100.0, soc_range=(0.0, 1.0)):
        """
        initialization a fusion model with parameters:
        v_range
        f_range
        i_max
        soc_range
        """
        print("Current num_lines_alive = ", current_data.num_lines)

        # build a model
        self.model = Model("case33")
        # sqr of node voltage
        self.v_sqr = self.model.variable(
            "v_sqr", 33, Domain.inRange(v_range[0], v_range[1]))
        # sqr of line current
        self.i_sqr = self.model.variable(
            "i_sqr", 37, Domain.inRange(0.0, i_max ** 2))
        # frequent
        # self.frequency = self.model.variable(
        #     "frequency", Domain.inRange(f_range[0], f_range[1]))
        # SOC
        # self.soc=self.M.variable("soc",2,Domain.inRange(soc_range[0],soc_range[1]))
        # P_mt
        self.p_mt = self.model.variable("p_mt", 3)
        # Q_mt
        self.q_mt = self.model.variable("q_mt", 3)
        # load-shedding state , the first element fixed to 0
        self.load_shed = self.model.variable("load_shed", 32, Domain.binary())
        # breaker state
        self.alpha = self.model.variable(
            "alpha", 37, Domain.binary())
        # energized state of node
        self.epsilon = self.model.variable("epsilon", 33, Domain.binary())
        # target variable for the DIRECTED multi-commodity flow constraint
        self.beta = self.model.variable(
            "beta", 74, Domain.binary())
        # auxiliary variable for the DIRECTED multi-commodity flow constraint
        self.lam = self.model.variable("lam", 74, Domain.binary())
        '''
        fictitious flow for Directed Multi-commodity Flow constraint
        f_flow^{k}_{i,j} >=0   where k is k_th node_rootfree, (i,j) is lines including the failed ones
        '''
        self.f_flow = self.model.variable(
            "f_flow", [74, 32], Domain.greaterThan(0.0))
        # auxiliary variable for "ij" term for McCormick envelopes
        self.w1 = self.model.variable("w1", 33, Domain.greaterThan(0))
        # auxiliary variable for "jk" term for McCormick envelopes
        self.w2 = self.model.variable("w2", 30, Domain.greaterThan(0))
        # ES charge flag and discharge flag
        # self.charge_flag=self.M.variable("charge_flag",2,Domain.binary())
        # self.discharge_flag=self.M.variable("discharge_flag",2,Domain.binary())

        # forward activate & reactivate power are free
        self.p_forward = self.model.variable(
            "p_forward", 37)
        self.q_forward = self.model.variable(
            "q_forward", 37)
        # inject activate & reactivate power
        self.p_in = self.model.variable("p_in", 33)
        self.q_in = self.model.variable("q_in", 33)

        # big M
        self.bigM = 1e6

        pass

    def make_constraints(self, current_data: GridData):

        # Do some preparation
        # set mask matrix for SOMETHING_ij ,SOMETHING_jk , mask matrix of r\x\z^2
        self.mask_matrix_ij, self.mask_matrix_r, self.mask_matrix_x, self.mask_matrix_sqrz, self.mask_matrix_i = current_data.make_matrix(
            mode="ij_forward", current_state=False)
        self.mask_matrix_jk = current_data.make_matrix(
            mode="jk_forward", current_state=False)

        # set alpha of failed line =>0
        self.model.constraint(self.alpha.pick(current_data.list_fault_line_number),Domain.equalsTo(0))
        # set the line mask matrix "root-free" in ij\jk mode
        # self.mask_matrix_ij_rootfree = np.delete(self.mask_matrix_ij, 0, 0)
        # self.mask_matrix_jk_rootfree = np.delete(self.mask_matrix_jk, 0, 0)

        # set mask matrix for topological constraints in pairs mode
        self.mask_matrix_pairs = current_data.make_matrix(
            mode="pairs", current_state=False)

        # set the mask matrix "root-free" in pairs mode
        self.mask_matrix_pairs_rootfree = np.delete(
            self.mask_matrix_pairs, 0, 0)
        '''
        set the injection power
        # default set is:
        # substation = { node[1] } without any load
        # MT = { node[4]; node[8]; node[22] }
        # WT = { node[28]; node[33] }
        # PV = { node[8] ; node[14] }
        # power_in = power_{sub\MT\WT\PV}- power_load
        '''
        # for MT-installed nodes

        st_mt_pin = self.model.constraint(Expr.sub(
            Expr.add(Expr.sub(self.p_mt, Expr.mulElm(self.load_shed.pick([3, 7, 21]), current_data.list_Pload_MT.tolist(
            ))), current_data.list_Ppv_MT.tolist()), self.p_in.pick([3, 7, 21])), Domain.equalsTo(0.0))
        st_mt_qin = self.model.constraint(Expr.sub(
            Expr.add(Expr.sub(self.q_mt, Expr.mulElm(self.load_shed.pick([3, 7, 21]), current_data.list_Qload_MT.tolist(
            ))), current_data.list_Qpv_MT.tolist()), self.q_in.pick([3, 7, 21])), Domain.equalsTo(0.0))

        # for MT-free nodes
        # P_pv + P_wt - shed * P_load = P_in
        st_pin = self.model.constraint(Expr.sub(Expr.sub(Expr.add(
            Expr.mulElm(self.load_shed.pick(
                current_data.list_in_loadshed.tolist()), current_data.list_Pload.tolist()),
            self.p_in.pick(
                current_data.list_in.tolist())), current_data.list_Ppv.tolist()), current_data.list_Pwt.tolist()),
            Domain.equalsTo(0.0))
        st_qin = self.model.constraint(Expr.sub(Expr.sub(Expr.add(
            Expr.mulElm(self.load_shed.pick(
                current_data.list_in_loadshed.tolist()), current_data.list_Qload.tolist()),
            self.q_in.pick(
                current_data.list_in.tolist())), current_data.list_Qpv.tolist()), current_data.list_Qwt.tolist()),
            Domain.equalsTo(0.0))

        # Dist-Flow power constraints
        # Dist-Flow power constraints

        st_df_p = self.model.constraint(Expr.sub(
            Expr.sub(Expr.add(self.p_in, Expr.mul(self.mask_matrix_ij.tolist(), self.p_forward)),
                     Expr.mul(self.mask_matrix_r.tolist(), self.i_sqr)),
            Expr.mul(self.mask_matrix_jk.tolist(), self.p_forward)), Domain.equalsTo(0.0))
        st_df_q = self.model.constraint(Expr.sub(
            Expr.sub(Expr.add(self.q_in, Expr.mul(self.mask_matrix_ij.tolist(), self.q_forward)),
                     Expr.mul(self.mask_matrix_x.tolist(), self.i_sqr)),
            Expr.mul(self.mask_matrix_jk.tolist(), self.q_forward)), Domain.equalsTo(0.0))

        # Dist-Flow node voltage constraints
        # test=self.model.constraint(Expr.mul(50,Expr.sub(1,self.alpha.index(37))),Domain.equalsTo(0.0))

        for idx in range(0, current_data.num_lines):
            ij, r, x, z_sqr = current_data.lookup(idx, current_state=False)
            self.model.constraint(Expr.add(Expr.sub(Expr.add(Expr.sub(self.v_sqr.index(ij[0]-1), self.v_sqr.index(ij[1]-1)), Expr.mul(self.bigM, Expr.sub(
                1, self.alpha.index(idx)))), Expr.mul(2, Expr.add(Expr.mul(r, self.p_forward.index(idx)), Expr.mul(x, self.q_forward.index(idx))))), Expr.mul(z_sqr, self.i_sqr.index(idx))), Domain.greaterThan(0.0))
            self.model.constraint(Expr.add(Expr.sub(Expr.sub(Expr.sub(self.v_sqr.index(ij[0]-1), self.v_sqr.index(ij[1]-1)), Expr.mul(self.bigM, Expr.sub(
                1, self.alpha.index(idx)))), Expr.mul(2, Expr.add(Expr.mul(r, self.p_forward.index(idx)), Expr.mul(x, self.q_forward.index(idx))))), Expr.mul(z_sqr, self.i_sqr.index(idx))), Domain.lessThan(0.0))
            pass

        # R-SOC constraints
        for j in range(0, 33):
            for jk in np.nonzero(current_data.seek_neighbor(j + 1, mode="jk_forward", current_state=False))[0].tolist():
                self.model.constraint(Expr.mulElm([0.5, 1, 1, 1], Var.vstack(
                    Var.vstack(Var.vstack(self.v_sqr.index(j),
                                          self.i_sqr.index(jk)), self.p_forward.index(jk)),
                    self.q_forward.index(jk))), Domain.inRotatedQCone(4))
                pass
            pass
        '''
        Directed Multi Commodity Flow constraint (DMCF constraint)
        
        * f_flow^{k}_{i,j}    where k is k_th node_rootfree, (i,j) is lines including the failed ones
        
        st_DMCF_source: 1 unit commodity flow is sent from the root node to each load node
        st_DMCF_receive: each load ONLY accepts a specific kth commodity flow
        st_DMCF_reject: each load should REJECT the commodity flow for others 
        st_DMCF_aux1: relation between the auxiliary variable f_flow and lam  
        st_DMCF_aux2: sum(lam) = |N| - 1 
        st_DMCF_aux3: relation between the target variable beta and the auxiliary variable lam 
        '''
        # st_DMCF_source
        for k in range(32):
            self.model.constraint(Expr.sub(self.f_flow.index(
                0, k), self.f_flow.index(38, k)), Domain.equalsTo(-1))
            pass
        # st_DMCF_receive
        for k in range(32):
            self.model.constraint(Expr.sub(Expr.mul(current_data.seek_neighbor(k+2, mode="all_ij_forward", current_state=False).tolist(), self.f_flow.slice([0, k], [73, k])), Expr.mul(
                current_data.seek_neighbor(k+2, mode="all_jk_forward", current_state=False).tolist(), self.f_flow.slice([0, k], [73, k]))), Domain.equalsTo(1))
            pass
        # st_DMCF_reject
        for k in range(32):
            for j in range(32):
                if j != k:
                    self.model.constraint(Expr.sub(Expr.mul(current_data.seek_neighbor(j+2, mode="all_ij_forward", current_state=False).tolist(), self.f_flow.slice(
                        [0, k], [73, k])), Expr.mul(current_data.seek_neighbor(j+2, mode="all_jk_forward", current_state=False).tolist(), self.f_flow.slice([0, k], [73, k]))), Domain.equalsTo(0))
                    pass
                pass
            pass
        # st_DMCF_aux1
        for k in range(32):
            for idx in range(74):
                self.model.constraint(Expr.sub(self.f_flow.index(
                    idx, k), self.lam.index(idx)), Domain.lessThan(0))
                pass
            pass
        # st_DMCF_aux2
        self.model.constraint(Expr.sum(self.lam), Domain.equalsTo(32))
        # st_DMCF_aux3
        for idx in range(37):
            self.model.constraint(Expr.sub(Expr.add(self.lam.index(idx), self.lam.index(
                idx+37)), self.beta.index(idx)), Domain.equalsTo(0))
            pass

        '''
        Driver constraints of the energized state
        #NOTE:the (epsilon * alpha) term will be convexified by McCormick envelopes
        see https://optimization.mccormick.northwestern.edu/index.php/McCormick_envelopes
        we set W = epsilon * alpha
        with convexified by McCormick envelopes,
        W >= epsilon_min * alpha + epsilon * alpha_min - epsilon_min * alpha_min
        W >= epsilon_max * alpha + epsilon * alpha_max - epsilon_max * alpha_max
        W <= epsilon_max * alpha + epsilon * alpha_min - epsilon_max * alpha_min
        W <= epsilon * alpha_max + epsilon_min * alpha - epsilon_min * alpha_max
        
        * root(substation) and MT node is set as always energized => { 0, 3, 7, 21 }
        #TODO: it is HARD for Fusion to give a mask matrix for present the relationship between neighbors and two different W
        currently, we have to build the constraints manual :(
        - Fortunately, in case33bw, we need to handle 30 nodes 
        '''
        self.model.constraint(self.epsilon.pick(
            [0, 3, 7, 21]), Domain.equalsTo(0))
        # node 2
        self.model.constraint(Expr.sub(Expr.add(Expr.add(self.w1.index(0), self.w1.index(
            1)), self.w2.index(0)), Expr.mul(self.epsilon.index(1), 3)), Domain.lessThan(0))
        self.model.constraint(Expr.add(Expr.add(self.w1.index(
            0), self.w1.index(1)), self.w2.index(0)), Domain.greaterThan(3))

        self.model.constraint(Expr.sub(self.w1.index(
            0), self.alpha.index(1)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            0), self.epsilon.index(2)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(0), Expr.sub(Expr.add(
            self.alpha.index(1), self.epsilon.index(2)), 1)), Domain.greaterThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            1), self.alpha.index(17)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            1), self.epsilon.index(18)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(1), Expr.sub(Expr.add(
            self.alpha.index(17), self.epsilon.index(18)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            0), self.alpha.index(0)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            0), self.epsilon.index(0)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(0), Expr.sub(Expr.add(
            self.alpha.index(0), self.epsilon.index(0)), 1)), Domain.greaterThan(0))

        # node 3
        self.model.constraint(Expr.sub(Expr.add(Expr.add(self.w1.index(2), self.w1.index(
            3)), self.w2.index(1)), Expr.mul(self.epsilon.index(2), 3)), Domain.lessThan(0))
        self.model.constraint(Expr.add(Expr.add(self.w1.index(
            2), self.w1.index(3)), self.w2.index(1)), Domain.greaterThan(3))

        self.model.constraint(Expr.sub(self.w1.index(
            2), self.alpha.index(2)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            2), self.epsilon.index(3)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(2), Expr.sub(Expr.add(
            self.alpha.index(2), self.epsilon.index(3)), 1)), Domain.greaterThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            3), self.alpha.index(21)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            3), self.epsilon.index(22)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(3), Expr.sub(Expr.add(
            self.alpha.index(21), self.epsilon.index(22)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            1), self.alpha.index(1)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            1), self.epsilon.index(1)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(1), Expr.sub(Expr.add(
            self.alpha.index(1), self.epsilon.index(1)), 1)), Domain.greaterThan(0))

        # node 5
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(4), self.w2.index(
            2)), Expr.mul(self.epsilon.index(4), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            4), self.w2.index(2)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            4), self.alpha.index(4)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            4), self.epsilon.index(5)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(4), Expr.sub(Expr.add(
            self.alpha.index(4), self.epsilon.index(5)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            2), self.alpha.index(3)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            2), self.epsilon.index(3)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(2), Expr.sub(Expr.add(
            self.alpha.index(3), self.epsilon.index(3)), 1)), Domain.greaterThan(0))

        # node 6
        self.model.constraint(Expr.sub(Expr.add(Expr.add(self.w1.index(5), self.w1.index(
            6)), self.w2.index(3)), Expr.mul(self.epsilon.index(5), 3)), Domain.lessThan(0))
        self.model.constraint(Expr.add(Expr.add(self.w1.index(
            5), self.w1.index(6)), self.w2.index(3)), Domain.greaterThan(3))

        self.model.constraint(Expr.sub(self.w1.index(
            5), self.alpha.index(5)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            5), self.epsilon.index(6)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(5), Expr.sub(Expr.add(
            self.alpha.index(5), self.epsilon.index(6)), 1)), Domain.greaterThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            6), self.alpha.index(24)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            6), self.epsilon.index(25)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(6), Expr.sub(Expr.add(
            self.alpha.index(24), self.epsilon.index(25)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            3), self.alpha.index(4)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            3), self.epsilon.index(4)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(3), Expr.sub(Expr.add(
            self.alpha.index(4), self.epsilon.index(4)), 1)), Domain.greaterThan(0))

        # node 7
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(7), self.w2.index(
            4)), Expr.mul(self.epsilon.index(6), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            7), self.w2.index(4)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            7), self.alpha.index(6)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            7), self.epsilon.index(7)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(7), Expr.sub(Expr.add(
            self.alpha.index(6), self.epsilon.index(7)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            4), self.alpha.index(5)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            4), self.epsilon.index(5)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(4), Expr.sub(Expr.add(
            self.alpha.index(5), self.epsilon.index(5)), 1)), Domain.greaterThan(0))

        # node 9
        self.model.constraint(Expr.sub(Expr.add(Expr.add(self.w1.index(8), self.w1.index(
            9)), self.w2.index(5)), Expr.mul(self.epsilon.index(8), 3)), Domain.lessThan(0))
        self.model.constraint(Expr.add(Expr.add(self.w1.index(
            8), self.w1.index(9)), self.w2.index(5)), Domain.greaterThan(3))

        self.model.constraint(Expr.sub(self.w1.index(
            8), self.alpha.index(8)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            8), self.epsilon.index(8)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(8), Expr.sub(Expr.add(
            self.alpha.index(8), self.epsilon.index(8)), 1)), Domain.greaterThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            9), self.alpha.index(33)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            9), self.epsilon.index(14)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(9), Expr.sub(Expr.add(
            self.alpha.index(33), self.epsilon.index(14)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            5), self.alpha.index(7)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            5), self.epsilon.index(7)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(5), Expr.sub(Expr.add(
            self.alpha.index(7), self.epsilon.index(7)), 1)), Domain.greaterThan(0))

        # node 10
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(10), self.w2.index(
            6)), Expr.mul(self.epsilon.index(9), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            10), self.w2.index(6)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            10), self.alpha.index(9)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            10), self.epsilon.index(10)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(10), Expr.sub(Expr.add(
            self.alpha.index(9), self.epsilon.index(10)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            6), self.alpha.index(8)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            6), self.epsilon.index(8)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(6), Expr.sub(Expr.add(
            self.alpha.index(8), self.epsilon.index(8)), 1)), Domain.greaterThan(0))

        # node 11
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(11), self.w2.index(
            7)), Expr.mul(self.epsilon.index(10), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            11), self.w2.index(7)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            11), self.alpha.index(10)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            11), self.epsilon.index(11)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(11), Expr.sub(Expr.add(
            self.alpha.index(10), self.epsilon.index(11)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            7), self.alpha.index(9)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            7), self.epsilon.index(9)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(7), Expr.sub(Expr.add(
            self.alpha.index(9), self.epsilon.index(9)), 1)), Domain.greaterThan(0))

        # node 12
        self.model.constraint(Expr.sub(Expr.add(Expr.add(self.w1.index(12), self.w1.index(
            13)), self.w2.index(8)), Expr.mul(self.epsilon.index(11), 3)), Domain.lessThan(0))
        self.model.constraint(Expr.add(Expr.add(self.w1.index(
            12), self.w1.index(13)), self.w2.index(8)), Domain.greaterThan(3))

        self.model.constraint(Expr.sub(self.w1.index(
            12), self.alpha.index(11)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            12), self.epsilon.index(12)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(12), Expr.sub(Expr.add(
            self.alpha.index(11), self.epsilon.index(12)), 1)), Domain.greaterThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            13), self.alpha.index(34)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            13), self.epsilon.index(21)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(13), Expr.sub(Expr.add(
            self.alpha.index(34), self.epsilon.index(21)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            8), self.alpha.index(10)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            8), self.epsilon.index(10)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(8), Expr.sub(Expr.add(
            self.alpha.index(10), self.epsilon.index(10)), 1)), Domain.greaterThan(0))

        # node 13
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(14), self.w2.index(
            9)), Expr.mul(self.epsilon.index(12), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            14), self.w2.index(9)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            14), self.alpha.index(12)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            14), self.epsilon.index(13)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(14), Expr.sub(Expr.add(
            self.alpha.index(12), self.epsilon.index(13)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            9), self.alpha.index(11)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            9), self.epsilon.index(11)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(9), Expr.sub(Expr.add(
            self.alpha.index(11), self.epsilon.index(11)), 1)), Domain.greaterThan(0))

        # node 14
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(15), self.w2.index(
            10)), Expr.mul(self.epsilon.index(13), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            15), self.w2.index(10)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            15), self.alpha.index(13)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            15), self.epsilon.index(14)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(15), Expr.sub(Expr.add(
            self.alpha.index(13), self.epsilon.index(14)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            10), self.alpha.index(12)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            10), self.epsilon.index(12)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(10), Expr.sub(Expr.add(
            self.alpha.index(12), self.epsilon.index(12)), 1)), Domain.greaterThan(0))

        # node 15
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(16), self.w2.index(
            11)), Expr.mul(self.epsilon.index(14), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            16), self.w2.index(11)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            16), self.alpha.index(14)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            16), self.epsilon.index(15)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(16), Expr.sub(Expr.add(
            self.alpha.index(14), self.epsilon.index(15)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            11), self.alpha.index(13)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            11), self.epsilon.index(13)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(11), Expr.sub(Expr.add(
            self.alpha.index(13), self.epsilon.index(13)), 1)), Domain.greaterThan(0))

        # node 16
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(17), self.w2.index(
            12)), Expr.mul(self.epsilon.index(15), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            17), self.w2.index(12)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            17), self.alpha.index(15)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            17), self.epsilon.index(16)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(17), Expr.sub(Expr.add(
            self.alpha.index(15), self.epsilon.index(16)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            12), self.alpha.index(14)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            12), self.epsilon.index(14)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(12), Expr.sub(Expr.add(
            self.alpha.index(14), self.epsilon.index(14)), 1)), Domain.greaterThan(0))

        # node 17
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(18), self.w2.index(
            13)), Expr.mul(self.epsilon.index(16), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            18), self.w2.index(13)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            18), self.alpha.index(16)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            18), self.epsilon.index(17)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(18), Expr.sub(Expr.add(
            self.alpha.index(16), self.epsilon.index(17)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            13), self.alpha.index(15)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            13), self.epsilon.index(15)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(13), Expr.sub(Expr.add(
            self.alpha.index(15), self.epsilon.index(15)), 1)), Domain.greaterThan(0))

        # node 18
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(19), self.w2.index(
            14)), Expr.mul(self.epsilon.index(17), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            19), self.w2.index(14)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            19), self.alpha.index(35)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            19), self.epsilon.index(32)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(19), Expr.sub(Expr.add(
            self.alpha.index(35), self.epsilon.index(32)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            14), self.alpha.index(16)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            14), self.epsilon.index(16)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(14), Expr.sub(Expr.add(
            self.alpha.index(16), self.epsilon.index(16)), 1)), Domain.greaterThan(0))

        # node 19
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(20), self.w2.index(
            15)), Expr.mul(self.epsilon.index(18), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            20), self.w2.index(15)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            20), self.alpha.index(18)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            20), self.epsilon.index(19)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(20), Expr.sub(Expr.add(
            self.alpha.index(18), self.epsilon.index(19)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            15), self.alpha.index(17)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            15), self.epsilon.index(17)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(15), Expr.sub(Expr.add(
            self.alpha.index(17), self.epsilon.index(17)), 1)), Domain.greaterThan(0))

        # node 20
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(21), self.w2.index(
            16)), Expr.mul(self.epsilon.index(19), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            21), self.w2.index(16)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            21), self.alpha.index(18)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            21), self.epsilon.index(19)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(21), Expr.sub(Expr.add(
            self.alpha.index(18), self.epsilon.index(19)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            16), self.alpha.index(18)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            16), self.epsilon.index(18)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(16), Expr.sub(Expr.add(
            self.alpha.index(18), self.epsilon.index(18)), 1)), Domain.greaterThan(0))

        # node 21
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(22), self.w2.index(
            17)), Expr.mul(self.epsilon.index(20), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            22), self.w2.index(17)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            22), self.alpha.index(20)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            22), self.epsilon.index(21)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(22), Expr.sub(Expr.add(
            self.alpha.index(20), self.epsilon.index(21)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            17), self.alpha.index(19)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            17), self.epsilon.index(19)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(17), Expr.sub(Expr.add(
            self.alpha.index(19), self.epsilon.index(19)), 1)), Domain.greaterThan(0))

        # node 23
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(23), self.w2.index(
            18)), Expr.mul(self.epsilon.index(20), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            23), self.w2.index(18)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            23), self.alpha.index(22)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            23), self.epsilon.index(23)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(23), Expr.sub(Expr.add(
            self.alpha.index(20), self.epsilon.index(21)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            18), self.alpha.index(21)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            18), self.epsilon.index(2)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(18), Expr.sub(Expr.add(
            self.alpha.index(21), self.epsilon.index(2)), 1)), Domain.greaterThan(0))

        # node 24
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(24), self.w2.index(
            19)), Expr.mul(self.epsilon.index(22), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            24), self.w2.index(19)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            24), self.alpha.index(23)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            24), self.epsilon.index(24)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(24), Expr.sub(Expr.add(
            self.alpha.index(23), self.epsilon.index(24)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            19), self.alpha.index(22)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            19), self.epsilon.index(22)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(19), Expr.sub(Expr.add(
            self.alpha.index(22), self.epsilon.index(2)), 1)), Domain.greaterThan(0))

        # node 25
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(25), self.w2.index(
            20)), Expr.mul(self.epsilon.index(23), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            25), self.w2.index(20)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            25), self.alpha.index(36)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            25), self.epsilon.index(28)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(25), Expr.sub(Expr.add(
            self.alpha.index(36), self.epsilon.index(28)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            20), self.alpha.index(23)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            20), self.epsilon.index(23)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(20), Expr.sub(Expr.add(
            self.alpha.index(23), self.epsilon.index(23)), 1)), Domain.greaterThan(0))

        # node 26
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(26), self.w2.index(
            21)), Expr.mul(self.epsilon.index(24), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            26), self.w2.index(21)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            26), self.alpha.index(25)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            26), self.epsilon.index(26)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(26), Expr.sub(Expr.add(
            self.alpha.index(25), self.epsilon.index(26)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            21), self.alpha.index(24)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            21), self.epsilon.index(5)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(21), Expr.sub(Expr.add(
            self.alpha.index(24), self.epsilon.index(5)), 1)), Domain.greaterThan(0))

        # node 27
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(27), self.w2.index(
            22)), Expr.mul(self.epsilon.index(25), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            27), self.w2.index(22)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            27), self.alpha.index(26)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            27), self.epsilon.index(27)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(27), Expr.sub(Expr.add(
            self.alpha.index(26), self.epsilon.index(27)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            22), self.alpha.index(25)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            22), self.epsilon.index(25)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(22), Expr.sub(Expr.add(
            self.alpha.index(25), self.epsilon.index(25)), 1)), Domain.greaterThan(0))

        # node 28
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(28), self.w2.index(
            23)), Expr.mul(self.epsilon.index(26), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            28), self.w2.index(23)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            28), self.alpha.index(27)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            28), self.epsilon.index(28)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(28), Expr.sub(Expr.add(
            self.alpha.index(27), self.epsilon.index(28)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            23), self.alpha.index(26)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            23), self.epsilon.index(26)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(23), Expr.sub(Expr.add(
            self.alpha.index(26), self.epsilon.index(26)), 1)), Domain.greaterThan(0))

        # node 29
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(29), self.w2.index(
            24)), Expr.mul(self.epsilon.index(27), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            29), self.w2.index(24)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            29), self.alpha.index(28)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            29), self.epsilon.index(29)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(29), Expr.sub(Expr.add(
            self.alpha.index(28), self.epsilon.index(29)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            24), self.alpha.index(27)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            24), self.epsilon.index(27)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(24), Expr.sub(Expr.add(
            self.alpha.index(27), self.epsilon.index(27)), 1)), Domain.greaterThan(0))

        # node 30
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(30), self.w2.index(
            25)), Expr.mul(self.epsilon.index(28), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            30), self.w2.index(25)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            30), self.alpha.index(29)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            30), self.epsilon.index(30)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(30), Expr.sub(Expr.add(
            self.alpha.index(29), self.epsilon.index(30)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            25), self.alpha.index(28)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            25), self.epsilon.index(28)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(25), Expr.sub(Expr.add(
            self.alpha.index(28), self.epsilon.index(28)), 1)), Domain.greaterThan(0))

        # node 31
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(31), self.w2.index(
            26)), Expr.mul(self.epsilon.index(29), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            31), self.w2.index(26)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            31), self.alpha.index(30)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            31), self.epsilon.index(31)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(31), Expr.sub(Expr.add(
            self.alpha.index(30), self.epsilon.index(31)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            26), self.alpha.index(29)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            26), self.epsilon.index(29)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(26), Expr.sub(Expr.add(
            self.alpha.index(29), self.epsilon.index(29)), 1)), Domain.greaterThan(0))

        # node 32
        self.model.constraint(Expr.sub(Expr.add(self.w1.index(32), self.w2.index(
            27)), Expr.mul(self.epsilon.index(30), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w1.index(
            32), self.w2.index(27)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w1.index(
            32), self.alpha.index(31)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(
            32), self.epsilon.index(32)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w1.index(32), Expr.sub(Expr.add(
            self.alpha.index(31), self.epsilon.index(32)), 1)), Domain.greaterThan(0))

        self.model.constraint(Expr.sub(self.w2.index(
            27), self.alpha.index(30)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            27), self.epsilon.index(30)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(27), Expr.sub(Expr.add(
            self.alpha.index(30), self.epsilon.index(30)), 1)), Domain.greaterThan(0))

        # node 33
        self.model.constraint(Expr.add(Expr.add(self.w2.index(28), self.w2.index(
            29)), Expr.mul(self.epsilon.index(32), 2)), Domain.lessThan(0))
        self.model.constraint(Expr.add(self.w2.index(
            28), self.w2.index(29)), Domain.greaterThan(2))

        self.model.constraint(Expr.sub(self.w2.index(
            28), self.alpha.index(31)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            28), self.epsilon.index(30)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(28), Expr.sub(Expr.add(
            self.alpha.index(31), self.epsilon.index(30)), 1)), Domain.greaterThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            29), self.alpha.index(35)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(
            29), self.epsilon.index(17)), Domain.lessThan(0))
        self.model.constraint(Expr.sub(self.w2.index(29), Expr.sub(Expr.add(
            self.alpha.index(35), self.epsilon.index(17)), 1)), Domain.greaterThan(0))

        # output of MTs
        st_pmt = self.model.constraint(self.p_mt, Domain.inRange(
            [750e3, 125e3, 375e3], [3e6, 5e5, 1.5e6]))
        st_qmt = self.model.constraint(self.q_mt, Domain.inRange(
            [-1.5e6, -2.5e5, -1e6], [1.5e6, 2.5e5, 1e6]))

        # substaion
        # v_sub = 1.0 pu
        st_sub1 = self.model.constraint(
            self.v_sqr.index(0), Domain.equalsTo(12.66e3 ** 2))
        st_sub2 = self.model.constraint(
            self.p_in.index(0), Domain.lessThan(0.0))

    def make_objective(self, current_data: GridData):
        """
        set the optimization objective function

        obj = Minimize  Cost_blackout * P_blackout

        where:
        P_blackout = sum( (1-load_shed) * P_load)

        """

        obj_function = Expr.sum(Expr.mulElm(Expr.sub(np.ones(32).tolist(), self.load_shed),
                                            current_data.current_Pload.tolist()))
        self.model.objective("obj", ObjectiveSense.Minimize, obj_function)
        pass
    pass
