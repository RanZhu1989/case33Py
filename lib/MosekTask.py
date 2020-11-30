from mosek.fusion import *
from lib.GridData import *


class MosekTask:
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
        self.bigM = 50

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

        # set the mask matrix "root-free" in pairs modet
        self.mask_matrix_pairs_rootfree = np.delete(
            self.mask_matrix_pairs, 0, 0)

        # set the injection power
        # default set is:
        # substation = { node[1] } without any load
        # MT = { node[4]; node[8]; node[22] }
        # WT = { node[28]; node[33] }
        # PV = { node[8] ; node[14] }
        # power_in = power_{sub\MT\WT\PV}- power_load

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
        # st_st2= self.model.constraint(Expr.sub(Expr.sum(self.beta),self.beta.index(0)),Domain.equalsTo(1))
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
            [750e3, 125e3, 375e3], [3e6, 5e5, 1.5e6]))
        st_qmt = self.model.constraint(self.q_mt, Domain.inRange(
            [-1.5e6, -2.5e5, -1e6], [1.5e6, 2.5e5, 1e6]))

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
        self.model.solve()

        # write the log file
        if log == True:
            self.model.writeTask("./log/step_%d_test.opf" % (step+1))
            pass

        if debug == True:
            print(str(self.model.getPrimalSolutionStatus()))
            print(list(self.alpha.level()))
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
            current_data.map_lines(self.alpha.level())
        if debug==True:
            print(current_data.solution_breaker_state)
            print(np.nonzero(current_data.solution_breaker_state))
            pass

        pass

    pass


if __name__ == "__main__":
    # create data agent
    data_case33 = GridData()
    # set max step
    max_step = 10
    for s in range(0, max_step):
        # gather current data by moving a step
        print("Step = ", s)
        data_case33.make_step(step=s)
        problem = MosekTask(data_case33)
        problem.make_constraints(data_case33)
        problem.make_objective(data_case33)
        problem.solve(s, debug=True)
        pass

    pass
