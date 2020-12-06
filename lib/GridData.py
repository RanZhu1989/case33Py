from math import sqrt
from os import terminal_size
import sys
import pandas as pd
import numpy as np
np.set_printoptions(threshold=1e6)


class GridData:
    """
    data for solver and pandapower environment
    """

    def __init__(self, load="./data/load.csv",
                 pv="./data/pv.csv", wt="./data/wt.csv",
                 event="./data/event.csv", price="./data/price.csv",
                 grid="./data/grid.csv"):
        """
        load\pv\wt\event\price = data.csv patch
        """

        self.pathLoad = load
        self.pathPV = pv
        self.pathWT = wt
        self.pathEvent = event
        self.pathPrice = price
        self.current_gridPara = pd.read_csv(grid)
        self.num_lines = 37
        self.price_loss = 0.18
        self.price_blackout = 10
        # make a static topology table
        self.static_gridPara = self.current_gridPara.copy().reset_index()
        self.static_gridPara_half_forward = self.static_gridPara.copy().drop(
            labels=range(37, 74), axis=0).reset_index()
        self.static_gridPara_half_backward = self.static_gridPara.copy().drop(
            labels=range(0, 37), axis=0).reset_index()
        # init cash
        self.line_opened_cash = np.zeros(5)
        self.line_fault_cash = np.zeros(2)
        self.pin_cash = np.zeros(32)
        self.pmt_cash = np.zeros(3)
        self.qmt_cash = np.zeros(3)
        pass

    def make_step(self, step=0):
        '''
        read one row data from csv files 

        step = current row
        '''

        # read data from csv files of current step
        # type of load \ pv \ wt  = complex
        # type of event = int
        # type of price = float
        # type of grid parameters table = str

        self.current_load = 1000 * \
            np.loadtxt(self.pathLoad, dtype=complex,
                       delimiter=",", skiprows=step, max_rows=1)
        self.current_pv = 1000 * \
            np.loadtxt(self.pathPV, dtype=complex,
                       delimiter=",", skiprows=step, max_rows=1)
        self.current_wt = 1000 * \
            np.loadtxt(self.pathWT, dtype=complex,
                       delimiter=",", skiprows=step, max_rows=1)
        self.current_event = np.loadtxt(
            self.pathEvent, dtype=bool, delimiter=",", skiprows=step, max_rows=1)
        self.current_price = 1e-3 * \
            np.loadtxt(self.pathPrice, dtype=float,
                       delimiter=",", skiprows=step, max_rows=1)

        # make lists for injection power constraints
        # lists for MT-installed node

        self.list_Pload_MT = np.real(
            np.array([self.current_load[3], self.current_load[7], self.current_load[21]]))
        self.list_Qload_MT = np.imag(
            np.array([self.current_load[3], self.current_load[7], self.current_load[21]]))
        self.list_Ppv_MT = np.array([0, np.real(self.current_pv[7]), 0])
        self.list_Qpv_MT = np.array([0, np.imag(self.current_pv[7]), 0])

        # current P\Q loads

        self.current_Pload = np.real(self.current_load)
        self.current_Pload = np.delete(self.current_Pload, 0, 0)
        self.current_Qload = np.imag(self.current_load)
        self.current_Qload = np.delete(self.current_Qload, 0, 0)

        # lists for MT-free node

        self.list_Pload = self.current_load.copy()
        self.list_Pload = np.real(np.delete(self.list_Pload, [0, 3, 7, 21]))
        self.list_Qload = self.current_load.copy()
        self.list_Qload = np.imag(np.delete(self.list_Qload, [0, 3, 7, 21]))
        self.list_Ppv = np.zeros(29)
        self.list_Ppv[11] = np.real(self.current_pv[13])
        self.list_Qpv = np.zeros(29)
        self.list_Qpv[11] = np.imag(self.current_pv[13])
        self.list_Pwt = np.zeros(29)
        self.list_Pwt[25] = np.real(self.current_wt[27])
        self.list_Qwt = np.zeros(29)
        self.list_Qwt[25] = np.imag(self.current_wt[27])

        # index for power_in and load_shed variables

        self.list_in = np.arange(0, 33, dtype=int)
        self.list_in = np.delete(self.list_in, [0, 3, 7, 21])
        self.list_in_loadshed = np.arange(0, 32, dtype=int)
        self.list_in_loadshed = np.delete(self.list_in_loadshed, [3, 7, 21])

        # make a copy of current grid parameters table
        # updated the flag of line_fault
        self.copy_current_gridPara = self.current_gridPara.copy()
        self.copy_current_gridPara["line_fault"] = self.copy_current_gridPara["line_fault"] | np.tile(
            self.current_event, 2)

        # make a copy for directed search, forward and backward
        self.current_gridPara_half = self.copy_current_gridPara.copy()
        self.current_gridPara_half_forward = self.current_gridPara_half.drop(
            labels=range(37, 74), axis=0)
        self.current_gridPara_half_backward = self.current_gridPara_half.drop(
            labels=range(0, 37), axis=0)

        # remove rows of fault lines and then reset index after drop cows
        self.copy_current_gridPara = self.copy_current_gridPara[
            ~self.copy_current_gridPara["line_fault"].isin([True])].reset_index(drop=True)
        self.current_gridPara_half_forward = self.current_gridPara_half_forward[
            ~self.current_gridPara_half_forward["line_fault"].isin([True])].reset_index(drop=True)
        self.current_gridPara_half_backward = self.current_gridPara_half_backward[
            ~self.current_gridPara_half_backward["line_fault"].isin([True])].reset_index(drop=True)

        # get number of lines alive
        self.num_lines = self.current_gridPara_half_forward.shape[0]

        # make list of fault line
        self.list_fault_line = self.make_fault_list()
        self.list_fault_line_number = list(np.nonzero(self.current_event))[0]
        self.list_fault_line_number = self.list_fault_line_number.astype(
            np.int32)

        # get r_line alive
        self.list_r = self.current_gridPara_half_forward["r"]

        # Initializing optimizer results buffer
        self.solution_breaker_state = np.zeros(37)
        self.solution_mt_p = np.zeros(3)
        self.solution_mt_q = np.zeros(3)
        self.solution_loadshed = np.zeros(32)

        pass

    def seek_neighbor(self, node, mode="jk_forward", current_state=True):
        """
        set input node = j
        j -> row for matrix A
        This function helps to create the neighbor-relational matrix for constraints build

        # current_state = True => search the topology ignoring the filed lines
        # current_state = False => search the topology using the original network 

        **jk_forward .= j -> k , start node = j 
        +RETURN:
            (list)mask_row : row of mask matrix for constrant building


        **ij_forward .= i -> j , end node = j 
        +RETURN:
            (list)mask_row : row of mask matrix for constraints building
            (list)mask_r_row:row of mask matrix for Dist-Flow with multiply r value
            (list)mask_x_row:row of mask matrix for Dist-Flow with multiply x value
            (list)mask_sqrz_row:row of mask matrix for Dist-Flow with multiply z^2 value
            (list)mask_i_row:row of mask matrix for find the neighbor i node, which is constrained in (i,j) edge

        **all_ij_forward .=  find (i,j) in all (i,j)(j,i) pairs
          or
          all_jk_forward .=  find (j,k) in all (i,j)(j,i) pairs
        +RETURN:
            (list)mask_row : row of mask matrix for constraints building

        **pairs .= backward node <- j get (i,j) AND backward node -> j get (j,i)
        +RETURN:
            (list)mask_row : row of mask matrix for topology constraints building

        **count .= count the number of neighbor nodes of the input node j
        +RETURN:
            (int)num_neighbor : scalar


        """
        if current_state == True:
            num_cols = self.current_gridPara_half_forward.shape[0]
        else:
            num_cols = 37
            pass

        # print(num_cols)

        if mode == "jk_forward":
            if current_state == True:
                find_jk_forward = self.current_gridPara_half_forward[
                    self.current_gridPara_half_forward["node_i"].isin([node])]
            else:
                find_jk_forward = self.static_gridPara_half_forward[
                    self.static_gridPara_half_forward["node_i"].isin([node])]
                pass
            idx = find_jk_forward.index
            mask_row = np.zeros(num_cols, dtype=int)
            mask_row[list(idx)] = 1
            return mask_row

        if mode == "ij_forward":
            if current_state == True:
                find_ij_forward = self.current_gridPara_half_forward[
                    self.current_gridPara_half_forward["node_j"].isin([node])]
            else:
                find_ij_forward = self.static_gridPara_half_forward[
                    self.static_gridPara_half_forward["node_j"].isin([node])]
                pass
            idx = find_ij_forward.index
            mask_row = np.zeros(num_cols, dtype=int)
            mask_row_i = np.zeros(34, dtype=int)
            mask_row[list(idx)] = 1
            mask_r_row = np.zeros(num_cols, dtype=float)
            mask_x_row = mask_r_row.copy()
            mask_sqrz_row = mask_r_row.copy()
            mask_r_row[list(idx)] = list(find_ij_forward["r"])
            mask_x_row[list(idx)] = list(find_ij_forward["x"])
            mask_sqrz_row = list(
                map(lambda x, y: x ** 2 + y ** 2, mask_r_row, mask_x_row))
            mask_row_i[list(find_ij_forward["node_i"])] = 1
            # drop the first element
            mask_row_i = mask_row_i[1:]
            return mask_row, mask_r_row, mask_x_row, mask_sqrz_row, mask_row_i
        '''
        the "all_" mode often used to build Directed Multi Commodity Flow constraint
        in DNR task, some nodes should be ignored
        .iloc[[idx]]["node_i"]
        '''
        if mode == "all_ij_forward":
            mask_row = np.zeros(num_cols*2, dtype=int)
            if current_state == True:
                find_ij_forward = self.copy_current_gridPara[
                    self.copy_current_gridPara["node_j"].isin([node])]
            else:
                find_ij_forward = self.static_gridPara[
                    self.static_gridPara["node_j"].isin([node])]
                pass
            idx = find_ij_forward.index
            mask_row[list(idx)] = 1
            return mask_row

        if mode == "all_jk_forward":
            mask_row = np.zeros(num_cols*2, dtype=int)
            if current_state == True:
                find_ij_forward = self.copy_current_gridPara[
                    self.copy_current_gridPara["node_i"].isin([node])]
            else:
                find_ij_forward = self.static_gridPara[
                    self.static_gridPara["node_i"].isin([node])]
                pass
            pass
            idx = find_ij_forward.index
            mask_row[list(idx)] = 1
            return mask_row

        if mode == "pairs":

            if current_state == True:
                # search in "ij" table
                find_all_ij = self.copy_current_gridPara.loc[0:self.num_lines -
                                                             1][self.copy_current_gridPara.loc[0:self.num_lines-1]["node_i"].isin([node])]
            # search in "ji" table
                find_all_ji = self.copy_current_gridPara.loc[self.num_lines:2*self.num_lines -
                                                             1][self.copy_current_gridPara.loc[self.num_lines:2*self.num_lines-1]["node_i"].isin([node])]
            else:
                find_all_ij = self.static_gridPara.loc[0:self.num_lines -
                                                       1][self.static_gridPara.loc[0:self.num_lines-1]["node_i"].isin([node])]
                find_all_ji = self.static_gridPara.loc[self.num_lines:2*self.num_lines -
                                                       1][self.static_gridPara.loc[self.num_lines:2*self.num_lines-1]["node_i"].isin([node])]

            idx_ij = find_all_ij.index
            idx_ji = find_all_ji.index
            mask_row = np.zeros(num_cols * 2, dtype=int)
            mask_row[list(idx_ij)] = 1
            mask_row[list(idx_ji)] = 1
            return mask_row

        pass

    def make_matrix(self, mode="jk_forward", current_state=True):
        """
        # current_state = True => search the topology ignoring the filed lines
        # current_state = False => search the topology using the original network 

        build matrix for constrants:

        #both for jk\ij_forward mode
            res_line_matrix => sum(P_ij\Q_ij) for SOCP
        #ij_forward mode only
            res_r_matrix    => r_ij * P_ij
            res_x_matrix    => x_ij * Q_ij
            res_sqrz_matrix => z^2_ij * l_ij
            res_i_matrix    => v_i - v_j
        """
        if current_state == False:
            num_lines = 37
        else:
            num_lines = self.num_lines

        if mode == "pairs":
            res_line_matrix = np.zeros((33, num_lines * 2), dtype=int)
        else:
            res_line_matrix = np.zeros((33, num_lines), dtype=int)

        res_r_matrix = np.zeros((33, num_lines), dtype=float)
        res_x_matrix = np.zeros((33, num_lines), dtype=float)
        res_sqrz_matrix = np.zeros((33, num_lines), dtype=float)
        res_i_matrix = np.zeros((33, 33), dtype=int)

        for j in range(0, 33):
            if mode == "jk_forward":
                res_line_matrix[j, ] = self.seek_neighbor(
                    j + 1, "jk_forward", current_state)
                pass
            if mode == "ij_forward":
                res_line_matrix[j, ], res_r_matrix[j, ], res_x_matrix[j, ], res_sqrz_matrix[j, ], res_i_matrix[
                    j, ] = self.seek_neighbor(j + 1, "ij_forward", current_state)
                pass
            if mode == "pairs":
                res_line_matrix[j, ] = self.seek_neighbor(
                    j + 1, "pairs", current_state)
            pass

        if mode == "jk_forward" or mode == "pairs":
            return res_line_matrix

        if mode == "ij_forward":
            return res_line_matrix, res_r_matrix, res_x_matrix, res_sqrz_matrix, res_i_matrix

        pass

    def lookup(self, idx, mode="line_ij", current_state=True):
        """
        # current_state = True => search the topology ignoring the filed lines
        # current_state = False => search the topology using the original network 

        lookup the meanings of variables, including :
        1. [index] of variable.alpha[dim == linesalive] => pairs such as (1,2) 
        2. line parameters like r,x,z^2

        """
        if mode == "pairs":
            if current_state == True:
                return (self.copy_current_gridPara.iloc[[idx]]["node_i"].tolist()[0],
                        self.copy_current_gridPara.iloc[[idx]]["node_j"].tolist()[0])
            else:
                return (self.static_gridPara.iloc[[idx]]["node_i"].tolist()[0],
                        self.static_gridPara.iloc.iloc[[idx]]["node_j"].tolist()[0])
        if mode == "line_ij":
            if current_state == True:
                r = self.current_gridPara_half_forward.iloc[[idx]]["r"].tolist()[
                    0]
                x = self.current_gridPara_half_forward.iloc[[idx]]["r"].tolist()[
                    0]
                z_sqr = r**2+x**2
                return (self.current_gridPara_half_forward.iloc[[idx]]["node_i"].tolist()[0],
                        self.current_gridPara_half_forward.iloc[[idx]]["node_j"].tolist()[0]), r, x, z_sqr
            else:
                r = self.static_gridPara_half_forward.iloc[[idx]]["r"].tolist()[
                    0]
                x = self.static_gridPara_half_forward.iloc[[idx]]["r"].tolist()[
                    0]
                z_sqr = r**2+x**2
                return (self.static_gridPara_half_forward.iloc[[idx]]["node_i"].tolist()[0],
                        self.static_gridPara_half_forward.iloc[[idx]]["node_j"].tolist()[0]), r, x, z_sqr
        pass

    def make_fault_list(self):
        """
        make list of fault line
        """

        list_fault_line = list(np.nonzero(self.current_event))[0]
        res = []
        for idx in list_fault_line:
            find = (self.current_gridPara.drop(
                labels=range(37, 74), axis=0).iloc[[idx]]["node_i"].tolist()[0]-1,
                self.current_gridPara.drop(
                labels=range(37, 74), axis=0).iloc[[idx]]["node_j"].tolist()[0]-1)
            res.append(find)
        return res

    def map_lines(self, res_alpha):
        """
        map the variable alpha with dim(|linesAlive|) -> set breaker state
        closed = 1
        opened \ failed = 0 
        """

        self.solution_breaker_state[list(map(lambda x, y: int(
            x-y), list(self.current_gridPara_half_forward.loc[np.nonzero(res_alpha)]["line_no"]), np.ones(self.num_lines)))]=1
        # print("list=", list(map(lambda x, y: int(
        #     x-y), list(self.current_gridPara_half_forward.loc[np.nonzero(res_alpha)]["line_no"]), np.ones(self.num_lines))))
        pass

if __name__ == "__main__":
    '''
    TEST only
    '''
    d = GridData()
    d.make_step(1)
    for i in range(1, 34):
        print(d.seek_neighbor(i, mode="pairs"))
        print("\n")
        print(np.nonzero(d.seek_neighbor(i, mode="pairs")))
        print("\n")
