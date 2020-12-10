from lib.MosekTask import *
from lib.GridData import *
from lib.PandapowerTask import *

# create data agent
data_case33 = GridData()
# create IEEE33BW network
network = PandapowerTask()
data_case33.make_step(step=9)
problem = MosekDNR(data_case33)
problem.make_constraints(data_case33)
problem.make_objective(data_case33)
problem.solve(9, data_case33,log=True,debug=True)