#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import csv
import gamspy as gp

#%% Parameters initialization
N_limit = 170 # kg N/ha, direttiva nitrati
N_buffalo = 83 # kg N/ha, Determinazione del carico di azoto secondo il DM 7/4/06
N_strip	= 0.5
Land_spred_ha = 1
Land_cost = 589 # Euro/ha, Dati RICA, (Affitti passivi / sau in affitto)
p_fine = 0.3 # Probabilità di sanzione, 
Land_cost_0	= 0
Fine = 2582.28 # 516.46 ad euro 2582.28, Disciplina tecnica per la utilizzazione dei liquami zootecnici
Risk_aversion = 0.193 # Italian farmer risk aversion by elicitation using lotteries, source: https://doi.org/10.1002/aepp.13330

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
