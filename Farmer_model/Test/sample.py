#%%
from pulp import *
#create the prob variabel to contain the problem data
FeedType = ['Chicken','Beef','Mutton','Rice','Wheat','Gel']
cost = {'Chicken':0.013,'Beef':0.008,'Mutton':0.01,'Rice':0.002,'Wheat':0.005,'Gel':0.001}
prob = LpProblem('Cat', LpMinimize)
Chicken = {'Protein':0.1 , 'Fat':0.08, 'Fibre':0.001, 'Salt':0.002 }
Beef = {'Protein':0.2 , 'Fat':0.1, 'Fibre':0.005, 'Salt':0.005 }
Mutton = {'Protein': 0.150, 'Fat':0.110, 'Fibre':0.003, 'Salt':0.007 }
Rice = {'Protein': 0.0, 'Fat':0.010, 'Fibre':0.1, 'Salt':0.002 }
Wheat = {'Protein': 0.040, 'Fat':0.01, 'Fibre':0.150, 'Salt':0.008 }
Gel = {'Protein': 0.00, 'Fat':0.0, 'Fibre':0.0, 'Salt':0.0 }
#Decision variables
ingredientsvar = LpVariable.dict('ing',FeedType, 0)

#Objective function
# min 0.013x1 + 0.008x2
prob += lpSum(cost[i]*ingredientsvar[i] for i in FeedType), "Total Cost of Ingredients per can"

#Constraints
prob += lpSum(ingredientsvar[i] for i in FeedType) == 100, "PercentagesSum"
prob += lpSum(i['Protein'] * ingredientsvar[i] for i in FeedType )>=8.0,"ProteinRequirement"
prob+= lpSum(i['Fat'] for i in FeedType)  >= 6.0, "FatReq"
prob+=lpSum(i['Fibre'] for i in FeedType) <=2.0 , 'FibreReq'
prob+=lpSum(i['Salt'] for i in FeedType) <= 0.4, 'SaltReq'
prob.writeLP("WhiskasModel.lp")
prob.solve(CPLEX())
print('Status:', LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Total Cost of Ingredients per can = ", value(prob.objective))
#input()
# %%
"""
The Simplified Whiskas Model Python Formulation for the PuLP Modeller

Authors: Antony Phillips, Dr Stuart Mitchell  2007
"""

# Import PuLP modeler functions
from pulp import *

# Create the 'prob' variable to contain the problem data
prob = LpProblem("The Whiskas Problem", LpMinimize)

# The 2 variables Beef and Chicken are created with a lower limit of zero
x1 = LpVariable("ChickenPercent", 0, None, LpInteger)
x2 = LpVariable("BeefPercent", 0)
x3 = LpVariable('Mutton',0)
x4 = LpVariable('Rice',0)
x5 = LpVariable('Wheat',0)
x6 = LpVariable('Gel',0)

# The objective function is added to 'prob' first
prob += 0.013 * x1 + 0.008 * x2 + 0.01*x3+0.002*x4 +0.005 *x5 +0.001 *x6, "Total Cost of Ingredients per can"

# The five constraints are entered
prob += x1 + x2 + x3+ x4+ x5+ x6 == 100, "PercentagesSum"
prob += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
prob += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
prob += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
prob += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"

# The problem data is written to an .lp file
prob.writeLP("WhiskasModel.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Total Cost of Ingredients per can = ", value(prob.objective))

# %%
guests = 10
max_table_size =5
possible_tables = [tuple(c) for c in allcombinations(guests, max_table_size)]
x = LpVariable.dicts('table',possible_tables, )