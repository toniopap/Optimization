#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import csv
import gamspy as gp
import matplotlib.colors as colors
#%% Parameters initialization

N_limit = 170 # kg N/ha, direttiva nitrati
N_buffalo = 83 # kg N/ha, Determinazione del carico di azoto secondo il DM 7/4/06
N_strip_eff = 0.5 # Efficienza di rimozione dell'azoto
N_strip_eff_val = np.array([0.3, 0.5, 0.7]) # min, med, max
Land_spred_ha = 1
Land_cost = 589 # Euro/ha, Dati RICA, (Affitti passivi / sau in affitto)
Land_cost_to_right = 0.7 # fraction of the land cost to be paid in case of land rights
Land_rights = Land_cost * Land_cost_to_right
p_sanction = 0.3 # Probabilità di sanzione
p_sanction_val = np.arange(0.1,1,0.1)
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
# %% plot
def plot_results(results, par1 , par2, b = False):
    if b:
        i = 6
    else:
        i = 4
    if results[:,4].min() >= 0 and results[:,6].max() >= 0:
        vmin = -0.1
    else: 
        vmin = min(results[:,4].min(),results[:,6].min())
    if results[:,4].max() <= 0 and results[:,6].min() <= 0:
        vmax = +0.1
    else:
        vmax = max(results[:,4].max(),results[:,6].max())
    norm = colors.TwoSlopeNorm(vmin= vmin, vcenter=0 , vmax=vmax)
    results = np.array(results)
    plt.figure()
    plt.scatter(results[:,0],results[:,1],c=results[:,i],cmap='RdYlGn', norm=norm, s=150)
    plt.colorbar()
    plt.xlabel(par1)
    plt.ylabel(par2)
    if b:
        plt.title('With Biogas')
    else:
        plt.title('Without Biogas')
    if b:
        plt.savefig(('./Grafici/'+par1+'_'+par2+'_biogas.png'))
    else: 
        plt.savefig(('./Grafici/'+par1+'_'+par2+'.png'))
    plt.show()
# %%
def plot_results_heat():

    return
# %%

# %% Land rights and risk aversion
# Sensitivity analysis
results = []
for l in Land_cost_to_rights_val:
    l_ = l*Land_cost
    for r in Risk_aversion_val:
        results.append([l_,r,u_compliance(l_,r),u_non_compliance(p_sanction,Land_cost_0, r, Sanction),delta_compliance(l_,r,p_sanction,Land_cost_0, Sanction), u_compliance_biogas(l_,r,N_strip_eff),delta_compliance_b(l_,r,p_sanction,Land_cost_0, Sanction, N_strip_eff)])
results = np.array(results)
plot_results(results, 'Land rights', 'Risk aversion')
plot_results(results, 'Land rights', 'Risk aversion',1)

# %% Sanction and probability of sanction
results = []
for i in Sanction_val:
    for j in p_sanction_val:
        results.append([i,j,u_compliance(Land_rights, Risk_aversion),u_non_compliance(j,Land_cost_0, Risk_aversion, i),delta_compliance(Land_rights, Risk_aversion, j, Land_cost_0, i), u_compliance_biogas(Land_rights, Risk_aversion, N_strip_eff),delta_compliance_b(Land_rights, Risk_aversion, j, Land_cost_0, i, N_strip_eff)])
results = np.array(results)
plot_results(results, 'Sanction', 'Probability of sanction')
plot_results(results, 'Sanction', 'Probability of sanction',1)

# %%
results = []
for i in Sanction_val:
    for j in p_sanction_val:
        results.append([i,j,u_compliance(Land_rights, Risk_aversion),u_non_compliance(j,Land_cost_0, Risk_aversion, i),delta_compliance(Land_rights, Risk_aversion, j, Land_cost_0, i), u_compliance_biogas(Land_rights, Risk_aversion, N_strip_eff),delta_compliance_b(Land_rights, Risk_aversion, j, Land_cost_0, i, N_strip_eff)])
results = np.array(results)


    

# %%
results2 = np.zeros((len(Sanction_val),len(p_sanction_val)))
for i in range(len(Sanction_val)):
    for j in range(len(p_sanction_val)):
        results2[i,j]= delta_compliance(Land_rights, Risk_aversion, p_sanction_val[j], Land_cost_0, Sanction_val[i])

# Create some example data (replace with your data)
data = results2
# Plot the heatmap
def heatmap2d(arr: np.ndarray, xlabel, ylabel, x_param, y_param):

    plt.figure(figsize=(8,6))
    norm = colors.TwoSlopeNorm(vmin= -3, vcenter=0 , vmax=+5)
    img = plt.imshow(arr,aspect='auto', origin='lower', 
                 extent=[x_param[0], x_param[-1], y_param[0], y_param[-1]], 
                 interpolation='bilinear', cmap='RdYlGn', norm=norm)
    # Add colorbar
    plt.colorbar(img, label='Delta utility')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
heatmap2d(results2, 'Sanction', 'Probability of sanction',Sanction_val,p_sanction_val )
