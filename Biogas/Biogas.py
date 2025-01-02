#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pulp as lp
import csv
#%% Data
def load_data():
    dict = {}
    # Extract header (variable names) and data
    reader = csv.reader(open('biogas_data.csv'))
    header = next(reader)
    data = next(reader)
    for name,value in zip(header,data):
        globals()[name] =float(value)
        dict[name] = float(value)
    
    return dict
data = load_data()
def risk_averse_utility(value, risk_aversion):
    return value**(risk_aversion)

pi_1 = data['pi_1']
pi_0 = data['pi_0']
t_1 = data['t_1']
t_0= data['t_0']
dis_eff = data['dis_eff']
q_1 = data['q_1']
q_0 = data['q_0']
S_0 = data['S_0']
S_1 = data['S_1']
risk_aversion = data['risk_aversion']

#%% Model
# Create the 'prob' variable to contain the problem data
prob = lp.LpProblem("Biogas", lp.LpMaximize)
biogas_var = lp.LpVariable.dicts("var", data) 
# The objective function is added to 'prob' first

prob += lp.lpSum(pi_1*(S_1-biogas_var['t_1'])+(1-pi_1)*(S_0-biogas_var['t_0'])) , 'Objective Function'
# %%
# The constraints are added to 'prob'

# Agent’s participation constraint
prob += pi_1*biogas_var['t_1']**risk_aversion+(1-pi_1)*biogas_var['t_0']**risk_aversion - dis_eff >= 0, 'Agent’s participation constraint'

# The moral hazard incentive constraint
prob += pi_1*biogas_var['t_1']**risk_aversion+(1-pi_1)*biogas_var['t_0']**risk_aversion - dis_eff >= pi_0*biogas_var['t_1']**risk_aversion+(1-pi_0)*biogas_var['t_0']**risk_aversion, 'The moral hazard incentive constraint'
