#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import csv
import gamspy as gp

#%% Parameters initialization
N_strip	= 0.5
Land_spred_ha = 1	
Land_cost = 100
p_fine = 0.01
Land_cost_0	= 10
Fine = 500
Risk_aversion = 0.5

#%% The model
def u_compliance():
    return (Land_cost**Risk_aversion)*(-1)
def u_non_compliance():
    return p_fine*(-1)*(Land_cost_0+Fine)**Risk_aversion+(1-p_fine)*Land_cost_0**Risk_aversion*(-1)
def is_compliance():
    return u_compliance() > u_non_compliance()



def print_results():
    if is_compliance():
        print("Farmer comply")
    else:
        print("Farmer does not comply")
    
    print("Utility of compliance: ", u_compliance())
    print("Utility of non-compliance: ", u_non_compliance())
#%% Results
print_results()
