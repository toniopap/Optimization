#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
#%% Parameters initialization

N_limit = 170 # kg N/ha, direttiva nitrati
N_buffalo = 83 # kg N/ha, Determinazione del carico di azoto secondo il DM 7/4/06
N_strip_eff = 0.7 # Efficienza di rimozione dell'azoto
N_strip_eff_val = np.arange(0.5,1,0.1) # min, med, max
Land_spred_ha = 1
Land_cost = 589 # Euro/ha, Dati RICA, (Affitti passivi / sau in affitto)
Land_cost_to_right = 0.1 # fraction of the land cost to be paid in case of land rights
Land_rights = Land_cost * Land_cost_to_right
p_sanction = 0.3 # Probabilità di sanzione
p_sanction_val = np.arange(0.01,1,0.001)
Land_cost_to_rights_val = np.arange(0.001,0.2,0.001) #
Land_cost_0	= 0 #Land cost for non-compliace
Sanction = 2582.28  # 516.46 ad euro 5164.57 , Disciplina tecnica per la utilizzazione dei liquami zootecnici - Le sanzioni amministrative sono così graduate: a) in sede di prima violazione, la sanzione da applicare è pari alla sanzione minima prevista;b) in sede di seconda violazione, la sanzione da applicare è pari al 50% della sanzione massima prevista;c) in sede di terza violazione, la sanzione da applicare è pari al 75% della sanzione massima prevista;d) in sede di quarta e successiva violazione, la sanzione da applicare è pari alla sanzione massima prevista
Sanction_val = [516.46, 2582.28, 3873.43 , 5164.57] # min, 50%, 75%, max
Sanction_val2 = np.arange(516.46, 5164.57, 0.001) # min, 50%, 75%, max

Risk_aversion = 0.193 # Italian farmer risk aversion by elicitation using lotteries, source: https://doi.org/10.1002/aepp.13330 , 0.164 , 0.223
Risk_aversion_val = np.arange(0.164, 0.223, 0.001) # min, max
R_europe = 0.214 # Risk aversion for Europe
R_europe_val = np.arange(0.206, 0.223, 0.01) # min, max

#%% The model
def u_compliance(l_c, risk): # l_c: land cost, risk: risk aversion
    return (l_c**risk)*(-1)

def u_non_compliance(p_s, l_nc, risk, s): # p_s: probability of sanction, l_nc: land cost, risk: risk aversion, s: sanction
    return p_s*(-1)*(l_nc+s)**risk+(1-p_s)*(-1)*l_nc**risk

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
        plt.title('With Nitrate stripping technology')
    else:
        plt.title('Without Nitrate stripping')
    if b:
        plt.savefig(('./Grafici/'+par1+'_'+par2+'_biogas.png'))
    else: 
        plt.savefig(('./Grafici/'+par1+'_'+par2+'.png'))
    plt.show()

def heatmap2d(arr: np.ndarray, xlabel, ylabel, x_param, y_param, b = False, line = False):

    plt.figure(figsize=(8,6))
    norm = colors.TwoSlopeNorm(vmin= -2, vcenter=0 , vmax=+4)
    img = plt.imshow(arr,aspect='auto', origin='lower', 
                 extent=[ y_param[0], y_param[-1],x_param[0], x_param[-1]], 
                  interpolation= 'Bilinear', cmap='RdYlGn', norm=norm)
    # Add colorbar
    plt.colorbar(img, label='Delta utility')
    if line:
        zero_level = np.where(np.abs(arr) < 0.001)  # Indices where Delta Utility ≈ 0

        if zero_level[0].size > 0 and zero_level[1].size > 0:
            # Convert indices to corresponding parameter values
            x_vals = np.array([x_param[i] for i in zero_level[0]])  # Extract corresponding x values
            y_vals = np.array([y_param[j] for j in zero_level[1]])  # Extract corresponding y values

            # Sort points by x for interpolation
            sorted_indices = np.argsort(x_vals)
            x_vals_sorted = x_vals[sorted_indices]
            y_vals_sorted = y_vals[sorted_indices]
            

            # Interpolate to create a smooth line
            if len(x_vals_sorted) > 1:
                interp_func = interp1d(x_vals_sorted, y_vals_sorted, kind='linear', fill_value="extrapolate")
                x_interp = np.linspace(min(x_vals_sorted), max(x_vals_sorted), 100000)
                y_interp = interp_func(x_interp)
                plt.legend()
                plt.plot(y_interp, x_interp, label='Zero Δ Utility')
    
    if b:
        plt.title('With Nitrate stripping')
    else:
        plt.title('Without Nitrate stripping')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if b:
        plt.savefig(('./Grafici/'+xlabel+'_'+ylabel+'_biogas.png'))
    else:
        plt.savefig(('./Grafici/'+xlabel+'_'+ylabel+'.png'))
    plt.show()
def heatmap2d2(arr: np.ndarray, xlabel, ylabel, x_param, y_param, b = False):

    plt.figure(figsize=(8,6))
    norm = colors.TwoSlopeNorm(vmin= -1, vcenter=0 , vmax=+1)
    img = plt.imshow(arr,aspect='auto', origin='lower', 
                 extent=[ y_param[0], y_param[-1],x_param[0], x_param[-1]], 
                  interpolation= 'Bilinear', cmap='RdYlGn')
    # Add colorbar
    plt.colorbar(img, label='Delta utility')
    if b:
        plt.title('With Nitrate stripping')
    else:
        plt.title('Without Nitrate stripping')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if b:
        plt.savefig(('./Grafici/'+xlabel+'_'+ylabel+'_biogas.png'))
    else:
        plt.savefig(('./Grafici/'+xlabel+'_'+ylabel+'.png'))
    plt.show()
def deltab_heat(arr: np.ndarray, xlabel, ylabel, x_param, y_param):
    plt.figure(figsize=(8,6))
    img = plt.imshow(arr,aspect='auto', origin='lower', 
                 extent=[ y_param[0], y_param[-1],x_param[0], x_param[-1]], 
                  interpolation= 'Bilinear', cmap='Blues')
        # Add colorbar
    plt.colorbar(img, label='Difference of Delta Utilities')
    plt.title('Difference of Delta Utilities between the two scenarios')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(('./Grafici/'+xlabel+'_'+ylabel+'_diff.png'))
    plt.show()


def surface3d_plot(arr: np.ndarray, xlabel, ylabel, x_param, y_param, b=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate meshgrid for the surface plot
    X, Y = np.meshgrid(y_param, x_param)
    Z = arr

    # Define the colormap normalization
    norm = colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    cmap = plt.cm.RdYlGn

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, edgecolor='k', alpha=0.9)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Delta utility')

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Delta Utility')
    ax.set_title('With Biogas' if b else 'Without Biogas')

    # Save the figure
    filename = f'./Grafici/{xlabel}_{ylabel}' + ('_biogas.png' if b else '.png')
    plt.savefig(filename)

    # Show the plot
    plt.show()

# %% Disposal costs and risk aversion
# Sensitivity analysis
results = np.zeros((len(Land_cost_to_rights_val),len(Risk_aversion_val)))
resultsb  = np.zeros((len(Land_cost_to_rights_val),len(Risk_aversion_val)))
deltab  = np.zeros((len(Land_cost_to_rights_val),len(Risk_aversion_val)))
for i in range(len(Land_cost_to_rights_val)):
    for j in range(len(Risk_aversion_val)):
        L_r = Land_cost * Land_cost_to_rights_val[i]*0.3
        results[i,j]= delta_compliance(L_r, Risk_aversion_val[j], p_sanction, Land_cost_0, Sanction)
        resultsb[i,j]= delta_compliance_b(L_r, Risk_aversion_val[j], p_sanction, Land_cost_0, Sanction, N_strip_eff)
        deltab[i,j]= resultsb[i,j] - results[i,j]
# Plot the heatmap
lr = Land_cost_to_rights_val*Land_cost
heatmap2d(results, 'Risk aversion', 'Manure disposal cost', lr, Risk_aversion_val,0,1)
heatmap2d(resultsb, 'Risk aversion', 'Manure disposal cost', lr, Risk_aversion_val,1)
deltab_heat(deltab, 'Risk aversion', 'Manure disposal cost',Sanction_val,p_sanction_val)
# %% Sanction and probability of sanction
results2 = np.zeros((len(Sanction_val),len(p_sanction_val)))
resultsb  = np.zeros((len(Sanction_val),len(p_sanction_val)))
deltab = np.zeros((len(Sanction_val),len(p_sanction_val)))
for i in range(len(Sanction_val)):
    for j in range(len(p_sanction_val)):
        results2[i,j] = delta_compliance(Land_rights, Risk_aversion, p_sanction_val[j], Land_cost_0, Sanction_val[i])
        resultsb[i,j] = delta_compliance_b(Land_rights, Risk_aversion, p_sanction_val[j], Land_cost_0, Sanction_val[i], N_strip_eff)
        deltab[i,j]= resultsb[i,j] - results2[i,j]
# Plot the heatmap
heatmap2d(results2,  'Probability of sanction', 'Sanction',Sanction_val,p_sanction_val,0,1 )
heatmap2d(resultsb,  'Probability of sanction', 'Sanction',Sanction_val,p_sanction_val,1,1 )
deltab_heat(deltab, 'Probability of sanction', 'Sanction',Sanction_val,p_sanction_val)

# %% Risk aversion and probability of sanction
results = np.zeros((len(p_sanction_val),len(Risk_aversion_val)))
resultsb  = np.zeros((len(p_sanction_val),len(Risk_aversion_val)))
deltab = np.zeros((len(p_sanction_val),len(Risk_aversion_val)))

for i in range(len(Risk_aversion_val)):
    for j in range(len(p_sanction_val)):
        results[j,i]= delta_compliance(Land_rights, Risk_aversion_val[i], p_sanction_val[j], Land_cost_0, Sanction)
        resultsb[j,i]= delta_compliance_b(Land_rights, Risk_aversion_val[i], p_sanction_val[j], Land_cost_0, Sanction, N_strip_eff)
        deltab[j,i]= resultsb[j,i] - results[j,i]
# Plot the heatmap
heatmap2d(results, 'Risk aversion', 'Probability of sanction', p_sanction_val, Risk_aversion_val, )
heatmap2d(resultsb, 'Risk aversion', 'Probability of sanction', p_sanction_val,Risk_aversion_val, 1)
deltab_heat(deltab, 'Risk aversion', 'Probability of sanction', p_sanction_val,Risk_aversion_val)
# %%
# %% Risk aversion and Land_rights
results = np.zeros((len(Land_cost_to_rights_val),len(Risk_aversion_val)))
resultsb  = np.zeros((len(Land_cost_to_rights_val),len(Risk_aversion_val)))
for i in range(len(Land_cost_to_rights_val)):
    for j in range(len(Risk_aversion_val)):
        L_r = Land_cost * Land_cost_to_rights_val[i]
        results[i,j]= delta_compliance(L_r, Risk_aversion_val[j], p_sanction, Land_cost_0, Sanction)
        resultsb[i,j]= delta_compliance_b(L_r, Risk_aversion_val[j], p_sanction, Land_cost_0, Sanction, N_strip_eff)

#Plot the heatmap
heatmap2d(results, 'Risk aversion', 'Manure disposal cost',(Land_cost * Land_cost_to_rights_val), Risk_aversion_val)
heatmap2d(resultsb, 'Risk aversion', 'Manure disposal cost',(Land_cost * Land_cost_to_rights_val), Risk_aversion_val,1)
deltab_heat(resultsb-results, 'Risk aversion', 'Manure disposal cost',(Land_cost * Land_cost_to_rights_val), Risk_aversion_val)
# %% Nitrogen removal efficiency and probability of sanction
results = np.zeros((len(p_sanction_val),len(N_strip_eff_val)))
resultsb  = np.zeros((len(p_sanction_val),len(N_strip_eff_val)))
deltab = np.zeros((len(p_sanction_val),len(N_strip_eff_val)))
for i in range(len(p_sanction_val)):
    for j in range(len(N_strip_eff_val)):
        results[i,j]= delta_compliance(Land_rights, Risk_aversion, p_sanction_val[i], Land_cost_0, Sanction)
        resultsb[i,j]= delta_compliance_b(Land_rights, Risk_aversion, p_sanction_val[i], Land_cost_0, Sanction, N_strip_eff_val[j])
        deltab[i,j]= resultsb[i,j] - results[i,j]
# Plot the heatmap
heatmap2d(results, 'Nitrogen removal efficiency', 'Probability of sanction',p_sanction_val, N_strip_eff_val )
heatmap2d(resultsb, 'Nitrogen removal efficiency', 'Probability of sanction',p_sanction_val, N_strip_eff_val,1 )
deltab_heat(deltab, 'Nitrogen removal efficiency', 'Probability of sanction',p_sanction_val, N_strip_eff_val)
# %% Manure disposal cost and Nitrogen removal efficiency
results = np.zeros((len(Land_cost_to_rights_val),len(N_strip_eff_val)))
resultsb  = np.zeros((len(Land_cost_to_rights_val),len(N_strip_eff_val)))
for i in range(len(Land_cost_to_rights_val)):
    for j in range(len(N_strip_eff_val)):
        L_r = Land_cost * Land_cost_to_rights_val[i]
        results[i,j]= delta_compliance(L_r, Risk_aversion, p_sanction, Land_cost_0, Sanction)
        resultsb[i,j]= delta_compliance_b(L_r, Risk_aversion, p_sanction, Land_cost_0, Sanction, N_strip_eff_val[j])
# Plot the heatmap
heatmap2d2(results, 'Nitrogen removal efficiency', 'Manure disposal cost', (Land_cost * Land_cost_to_rights_val), N_strip_eff_val)
heatmap2d2(resultsb, 'Nitrogen removal efficiency', 'Manure disposal cost', (Land_cost * Land_cost_to_rights_val), N_strip_eff_val,1)
deltab_heat(resultsb-results, 'Nitrogen removal efficiency', 'Manure disposal cost', (Land_cost * Land_cost_to_rights_val), N_strip_eff_val)


# %%
