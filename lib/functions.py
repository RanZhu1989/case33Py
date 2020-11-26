import numpy as np
import pandas as pd


class GridData:
    """
    data preprocess for solver
    """
    def __init__(self,load=".\\data\\load.csv",
                 pv=".\\data\\pv.csv",wt=".\\data\\wt.csv",
                 event=".\\data\\event.csv",price=".\\data\\price.csv",
                 grid=".\\data\\grid.csv"):
        
        """
        load\pv\wt\event\price = data.csv patch
        """
        
        self.pathLoad=load
        self.pathPV=pv
        self.pathWT=wt
        self.pathEvent=event
        self.pathPrice=price
        self.current_gridPara=pd.read_csv(grid)
        self.num_lines=37
        self.price_loss=0.18
        self.price_blackout=10
        
        pass
    
    def make_step(self,step=0):
        
        '''
        read one row data from csv files 
        
        step = current row
        '''        
        
        # read data from csv files of current step
        # type of load \ pv \ wt  = complex
        # type of event = int
        # type of price = float
        # type of grid parameters table = str 
        
        self.current_load=np.loadtxt(self.pathLoad,dtype=complex,delimiter=",",skiprows=step,max_rows=1)
        self.current_pv=np.loadtxt(self.pathPV,dtype=complex,delimiter=",",skiprows=step,max_rows=1)
        self.current_wt=np.loadtxt(self.pathWT,dtype=complex,delimiter=",",skiprows=step,max_rows=1)
        self.current_event=np.loadtxt(self.pathEvent,dtype=bool,delimiter=",",skiprows=step,max_rows=1)
        self.current_price=np.loadtxt(self.pathPrice,dtype=float,delimiter=",",skiprows=step,max_rows=1)
        
        # make lists for injection power constraints
        # lists for MT-installed node
        
        self.list_Pload_MT=np.real(np.array([self.current_load[3],self.current_load[7],self.current_load[21]]))
        self.list_Qload_MT=np.imag(np.array([self.current_load[3],self.current_load[7],self.current_load[21]]))
        self.list_Ppv_MT=np.array([0,np.real(self.current_pv[7]),0])
        self.list_Qpv_MT=np.array([0,np.imag(self.current_pv[7]),0])
        
        # lists for MT-free node
        
        self.list_Pload=self.current_load.copy()
        self.list_Pload=np.real(np.delete(self.list_Pload,[0,3,7,21]))
        self.list_Qload=self.current_load.copy()
        self.list_Qload=np.imag(np.delete(self.list_Qload,[0,3,7,21]))
        self.list_Ppv=np.zeros(29)
        self.list_Ppv[11]=np.real(self.current_pv[13])
        self.list_Qpv=np.zeros(29)
        self.list_Qpv[11]=np.imag(self.current_pv[13])
        
        #index for power_in and load_shed variables 
        
        self.list_in=np.arange(0,33,dtype=int)
        
        self.list_in=np.delete(self.list_in,[0,3,7,21])
        
        self.list_in_loadshed=np.arange(0,32,dtype=int)
        
        self.list_in_loadshed=np.delete(self.list_in_loadshed,[2,6,20])
        
        # make a copy of current grid parameters table
        # updated the flag of line_fault
        self.copy_current_gridPara=self.current_gridPara.copy()
        self.copy_current_gridPara["line_fault"]=self.copy_current_gridPara["line_fault"] | np.tile(self.current_event,2)
  
        # make a copy for directed search, forward and backward
        self.current_gridPara_half=self.copy_current_gridPara.copy()
        self.current_gridPara_half_forward=self.current_gridPara_half.drop(labels=range(37,74),axis=0)
        self.current_gridPara_half_backward=self.current_gridPara_half.drop(labels=range(0,37),axis=0)
        
        # remove rows of fault lines and then reset index after drop cows
        self.copy_current_gridPara=self.copy_current_gridPara[~self.copy_current_gridPara["line_fault"].isin([True])].reset_index(drop=True)
        self.current_gridPara_half_forward=self.current_gridPara_half_forward[
            ~self.current_gridPara_half_forward["line_fault"].isin([True])].reset_index(drop=True)
        self.current_gridPara_half_backward=self.current_gridPara_half_backward[
            ~self.current_gridPara_half_backward["line_fault"].isin([True])].reset_index(drop=True)
        
        #get number of lines alive
        self.num_lines=self.current_gridPara_half_forward.shape[0]
        pass
    
    def seek_neighbor(self,node,mode="jk_forward"):
        """
        set input node = j
        j -> row for matrix A
        This function helps to create the neighbor-relational matrix for constraints build
        
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
            
        **backward .= backward node <- j
        +RETURN:
            (list)mask_row : row of mask matrix for constraints building
        
        **pairs .= backward node <- j  *and* backward node -> j
        +RETURN:
            (list)mask_row : row of mask matrix for topology constraints building
        
        """
    
        num_cols=self.current_gridPara_half_forward.shape[0]
        
        if mode=="jk_forward":
            find_jk_forward=self.current_gridPara_half_forward[self.current_gridPara_half_forward["node_i"].isin([node])]
            idx=find_jk_forward.index
            mask_row=np.zeros(num_cols,dtype=int)
            mask_row[list(idx)]=1
            return mask_row
        
        if mode=="ij_forward":
            find_ij_forward=self.current_gridPara_half_forward[self.current_gridPara_half_forward["node_j"].isin([node])]
            idx=find_ij_forward.index
            mask_row=np.zeros(num_cols,dtype=int)
            mask_row_i=np.zeros(34,dtype=int)
            mask_row[list(idx)]=1
            mask_r_row=np.zeros(num_cols,dtype=float)
            mask_x_row=mask_r_row.copy()
            mask_sqrz_row=mask_r_row.copy()
            mask_r_row[list(idx)]=list(find_ij_forward["r"])
            mask_x_row[list(idx)]=list(find_ij_forward["x"])
            mask_sqrz_row=list(map(lambda x,y: x**2 + y**2, mask_r_row,mask_x_row))
            mask_row_i[list(find_ij_forward["node_i"])]=1
            #drop the first element
            mask_row_i=mask_row_i[1:]
            return mask_row,mask_r_row,mask_x_row,mask_sqrz_row,mask_row_i
        
        if mode=="backward":
            find_backward=self.current_gridPara_half_backward[self.current_gridPara_half_backward["node_j"].isin([node])]
            pass
        
        if mode=="pairs":
            find_all_ij=self.current_gridPara[self.current_gridPara["node_j"].isin([node])]
            find_all_ji=self.current_gridPara[self.current_gridPara["node_i"].isin([node])]
            idx_ij=find_all_ij.index
            idx_ji=find_all_ji.index
            mask_row=np.zeros(num_cols*2,dtype=int)
            mask_row[list(idx_ij)]=1
            mask_row[list(idx_ji)]=1
            return mask_row
        
        pass

    def make_matrix(self,mode="jk_forward"):
        
        """
        build matrix for constrants:
        
        #both for jk\ij_forward mode
            res_line_matrix => sum(P_ij\Q_ij) for SOCP
        #ij_forward mode only
            res_r_matrix    => r_ij * P_ij
            res_x_matrix    => x_ij * Q_ij
            res_sqrz_matrix => z^2_ij * l_ij
            res_i_matrix    => v_i - v_j
        """
        
        if mode=="pairs":
            res_line_matrix=np.zeros((33,self.num_lines*2),dtype=int)
        else:
            res_line_matrix=np.zeros((33,self.num_lines),dtype=int)
      
        res_r_matrix=np.zeros((33,self.num_lines),dtype=float)
        res_x_matrix=np.zeros((33,self.num_lines),dtype=float)
        res_sqrz_matrix=np.zeros((33,self.num_lines),dtype=float)
        res_i_matrix=np.zeros((33,33),dtype=int)
        
        for j in range(0,33):
            if mode=="jk_forward":
                res_line_matrix[j,]=self.seek_neighbor(j+1,"jk_forward")
                pass
            if mode=="ij_forward":
                res_line_matrix[j,],res_r_matrix[j,],res_x_matrix[j,],res_sqrz_matrix[j,],res_i_matrix[j,]=self.seek_neighbor(j+1,"ij_forward")
                pass
            if mode=="pairs":
                res_line_matrix[j,]=self.seek_neighbor(j+1,"pairs")
            pass
        
        if mode=="jk_forward"or mode=="pairs":
            return res_line_matrix
        
        if mode=="ij_forward":
            return res_line_matrix,res_r_matrix,res_x_matrix,res_sqrz_matrix,res_i_matrix
        
        pass     
        
    pass







if __name__=="__main__":
    '''
    TEST only
    '''
    d=GridData()
    d.make_step(7)
    l,r,x,z,i=d.make_matrix(mode="ij_forward")
    print(l)
    print(d.make_matrix(mode="jk_forward").shape)
    print(d.make_matrix(mode="pairs").shape)
    print(d.list_in_loadshed)
    print(type(d.current_price.tolist()[0]))
 