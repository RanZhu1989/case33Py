from matplotlib.pyplot import pause
from lib.MosekTask import *
from lib.GridData import *
from lib.PandapowerTask import *

# create data agent
data_case33 = GridData()
# create IEEE33BW network
network = PandapowerTask()
network.init_output()
# set start step
start_step= 0
# set end step
end_step = 25
for s in range(start_step, end_step):
    # gather current data by moving a step
    print("Step = ", s )
    data_case33.make_step(step=s)
    problem = MosekOPF(data_case33)
    problem.make_constraints(data_case33)
    problem.make_objective(data_case33)
    problem.solve(s, data_case33,log=False,debug=True)
    network.set_parameters(data_case33)
    if s==start_step:
        network.init_cache(data_case33)
    network.render(data_case33,plot=False,res_print=True,wait_time=5,logger=True)
    network.exp_out2file(data_case33)
    network.make_cache(data_case33)
    network.reset()
    pass
