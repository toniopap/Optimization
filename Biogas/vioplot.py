#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joypy
import matplotlib.cm as cm
# Step 1: Prepare the data
data = {
    'Category': [1, 2, 3, 4, 5, 6, 7],
    'Count1': [69, 77, 152, 158, 155, 136, 154],
    'Count2': [65,60,130,123,149,148,226],
    'Count3': [67,	83,	123,	196,	149,	116,	167],
    'Count4': [84,	80,	141,	159,	168,	137,	132],
    'Count5': [72,	68,	127,	159,	172,	150,	153],
    'Count6': [43,	62,	138,	140,	165,	151,	202],
    'Count7': [56,	67,	143,	237,	143,	142,	113],
    'Categorys': ['1-Dis', '2', '3', '4', '5', '6', '7 - acc'],

}
Satisfier = [[78,	76, 71 ,    68, 75,	89, 51, 55,	46],
[109,   113,	105,    86, 97, 113,    73, 63, 73],
[83,	84,	87,	92,	89,	77,	85,	74,	71],
[201,	215,	216,	199,	215,	203,	161,	156,	128],
[215,	223,	217,	214,	214,	200,	252,	285,	283],
[117,	118,	116,	147,	108,	126,	161,	163,	169],
[98,    72,     89,     95,     103,	93, 	118,	105,	131]
]
df_sat = pd.DataFrame(Satisfier)
#%%
hhh = []
expanded_data = []
for i in range(1,8):
    expanded_data = []
    for category, count in zip(data['Category'], data['Count'+str(i)]):
        expanded_data.extend([category] * count)
    hhh.append(expanded_data)


df = pd.DataFrame(hhh).T
df.columns = ['DS - 1', 'DS - 2', 'DS - 3', 'DS - 4', 'DS - 5', 'DS - 6', 'DS - 7']
col2 = [
'Human beings',
'Role of the state',
'Decisions of the society',
'Willingness of individuals to limit their freedom',
'Potential of consumption corridors',
'Meaning to limit consumption',
'Economic impact of consumption corridors']
#%%
hhh = []
expanded_data = []


for i in range(9):
    expanded_data = []
    for j in range(7):
        expanded_data.extend([j+1] * df_sat[i][j])
    hhh.append(expanded_data)
df_s_p = pd.DataFrame(hhh).T

# joy satisfier
fig, ax = joypy.joyplot(df_s_p, kind = "kde",fade = True,colormap = cm.Paired, linecolor = "white")


#%%
# Step 2: Plot the data
fig, ax = joypy.joyplot(df, kind = "kde",fade = True,colormap = cm.Paired, linecolor = "white")



# %%
# bar
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 12), sharex=True)
colors = plt.cm.Set2.colors
for i, ax in enumerate(axes):

    ax.bar(data['Category'], data['Count'+str((i+1))], alpha=0.5, 
           edgecolor=colors[i % len(colors)], color=colors[i % len(colors)],width=1)
   
    ax.set_ylim(0, 250)
    ax.set_yticks([])
    ax.annotate('DS - ' + str((i + 1)), xy=(0, 0), xytext=(-0.5, 0), 
                xycoords='axes fraction', textcoords='offset points', 
                ha='right', va='center', rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if i != 6:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.spines['bottom'].set_color(colors[i % len(colors)])
plt.tight_layout()
plt.show()
# %%
