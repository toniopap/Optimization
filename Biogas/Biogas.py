#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import csv
import gamspy as gp

#%% Parameters initialization
N_limit = 170 # kg N/ha, direttiva nitrati
N_buffalo = 83 # kg N/ha, Determinazione del carico di azoto secondo il DM 7/4/06
N_strip_eff = 0.5 # Efficienza di rimozione dell'azoto
N_strip_eff_val = np.array([0.3, 0.5, 0.7]) # min, med, max
Land_spred_ha = 1
Land_cost = 589 # Euro/ha, Dati RICA, (Affitti passivi / sau in affitto)
Land_cost_to_right = 0.7 # fraction of the land cost to be paid in case of land rights
Land_rights = Land_cost * Land_cost_to_right
p_sanction = 0.3 # Probabilità di sanzione, 
Land_cost_to_rights_val = np.arange(0.1,1,0.1) #
Land_cost_0	= 0
Sanction = 2582.28  # 516.46 ad euro 5164.57 , Disciplina tecnica per la utilizzazione dei liquami zootecnici - Le sanzioni amministrative sono così graduate: a) in sede di prima violazione, la sanzione da applicare è pari alla sanzione minima prevista;b) in sede di seconda violazione, la sanzione da applicare è pari al 50% della sanzione massima prevista;c) in sede di terza violazione, la sanzione da applicare è pari al 75% della sanzione massima prevista;d) in sede di quarta e successiva violazione, la sanzione da applicare è pari alla sanzione massima prevista
Sanction_val = [516.46, 2582.28, 3873.43 , 5164.57] # min, 50%, 75%, max

Risk_aversion = 0.193 # Italian farmer risk aversion by elicitation using lotteries, source: https://doi.org/10.1002/aepp.13330 , 0.164 , 0.223
Risk_aversion_val = np.arange(0.164, 0.223, 0.01) # min, max


#%% The model
def u_compliance(l_c, risk): # l_c: land cost, risk: risk aversion
    return (l_c**risk)*(-1)

def u_non_compliance(p_s, l_nc, risk, s): # p_s: probability of sanction, l_nc: land cost, risk: risk aversion, s: sanction
    return p_s*(-1)*(l_nc+s)**risk+(1-p_s)*l_nc**risk*(-1)

def u_compliance_biogas(l_c, risk, N_eff): # l_c: land cost, risk: risk aversion, N_strip_eff: nitrogen removal efficiency
    return ((l_c*(1-N_eff))**risk)*(-1)

def is_compliance(l_c,risk, p_s, l_nc, s):
    return u_compliance(l_c,risk) > u_non_compliance(p_s, l_nc, risk, s)

def delta_compliance(l_c,risk, p_s, l_nc, s):
    return u_compliance(l_c,risk) - u_non_compliance(p_s, l_nc, risk, s)

def delta_compliance_b(l_c,risk, p_s, l_nc, s,N_eff):
    return u_compliance_biogas(l_c,risk,N_eff) - u_non_compliance(p_s, l_nc, risk, s)

def is_compliance_b(l_c,risk, p_s, l_nc, s):
    return u_compliance_biogas(l_c,risk) > u_non_compliance(p_s, l_nc, risk, s)



#%% Results
# Average values
print (u_compliance(Land_rights, Risk_aversion))
print (u_non_compliance(p_sanction,Land_cost_0, Risk_aversion, Sanction))
print(u_compliance_biogas(Land_rights, Risk_aversion, N_strip_eff))
print(delta_compliance(Land_rights, Risk_aversion, p_sanction, Land_cost_0, Sanction))
print(delta_compliance_b(Land_rights, Risk_aversion, p_sanction, Land_cost_0, Sanction, N_strip_eff))

# %% Land rights and risk aversion
# Sensitivity analysis
results = []
for l in Land_cost_to_rights_val:
    l_ = l*Land_cost
    for r in Risk_aversion_val:
        results.append([l_,r,u_compliance(l_,r),u_non_compliance(p_sanction,Land_cost_0, r, Sanction),delta_compliance(l_,r,p_sanction,Land_cost_0, Sanction), u_compliance_biogas(l_,r,N_strip_eff),delta_compliance_b(l_,r,p_sanction,Land_cost_0, Sanction, N_strip_eff)])



# %% plot
results = np.array(results)
plt.figure()
plt.scatter(results[:,0],results[:,1],c=results[:,2],cmap='coolwarm')
plt.colorbar()



# %%
