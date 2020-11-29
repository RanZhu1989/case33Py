from os import terminal_size
from pandapower import networks
from lib.MosekTask import *
from lib.GridData import *
from lib.PandapowerTask import *

# create data agent
data_case33 = GridData()
# create IEEE33BW network
network = PandapowerTask()
# set max step
max_step = 10
for s in range(0, max_step):
    # gather current data by moving a step
    data_case33.make_step(step=s)
    problem = MosekTask(data_case33)
    problem.make_constraints(data_case33)
    problem.make_objective(data_case33)
    problem.solve(s, data_case33, debug=True)
    network.set_parameters(data_case33)
    network.render(data_case33)
    network.reset()
    pass
