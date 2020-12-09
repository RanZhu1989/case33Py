from lib.MosekTask import *
from lib.GridData import *
from lib.PandapowerTask import *

# create data agent
data_case33 = GridData()
# create IEEE33BW network
network = PandapowerTask()
# set start step
start_step= 0
# set end step
end_step = 8
for s in range(start_step, end_step):
    # gather current data by moving a step
    print("Step = ", s )
    data_case33.make_step(step=s)
    problem = MosekOPF(data_case33)
    problem.make_constraints(data_case33)
    problem.make_objective(data_case33)
    problem.solve(s, data_case33,log=False,debug=False)
    network.set_parameters(data_case33)
    network.render(data_case33,plot=True)
    if s>0:
        network.exp_out2file(data_case33)    
    network.make_cash(data_case33)
    network.reset()
    pass
