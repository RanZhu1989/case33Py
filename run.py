from task import *

# create data agent
data_case33 = GridData()
# set max step
max_step = 10

for s in range(0, max_step):
    # gather current data by moving a step
    data_case33.make_step(step=s)
    problem = SOCPTask(data_case33)
    problem.make_constraints(data_case33)
    problem.make_objective(data_case33)
    problem.solve()
    pass
