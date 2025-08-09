# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:03:19 2024

@author: Chen Min
"""


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle


import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


config = {
    "font.family":'Times New Roman',  
    "font.size": 16,
    "savefig.dpi": 300  
}
plt.rcParams.update(config) 




#%% significant difference


def data_group():

    with open('./results/C_all_mean.pkl', 'rb') as file_obj:
        C_all_mean = pickle.load(file_obj)
    
    C_all_mean = np.swapaxes(C_all_mean, 0, 1) 

    with open('./results/tum_time_diff_mean.pkl', 'rb') as file_obj:
        tum_time_diff = pickle.load(file_obj)
    

    good_p = np.array(np.where(tum_time_diff<0.4)).T #0.4s T<0.4s
    middle_p = np.array(np.where((tum_time_diff>0.4) & (tum_time_diff<0.55))).T # 0.4s<T<0.55
    bad_p = np.array(np.where(tum_time_diff>0.55)).T  # T>0.55s
    
    C_good = C_all_mean[:,good_p[:,0],good_p[:,1],:,:,:,:] 
    C_middle = C_all_mean[:,middle_p[:,0],middle_p[:,1],:,:,:,:]
    C_bad = C_all_mean[:,bad_p[:,0],bad_p[:,1],:,:,:,:] 
  
  
    return C_good, C_middle, C_bad



def anova_analysis(data):
    
    df_data = pd.DataFrame(data).T
    df_data.columns = ['G', 'M', 'B'] 
    df_data_melt = df_data.melt()
    df_data_melt = df_data_melt.dropna() 
    df_data_melt.columns = ['Class','Index']
    
    model = ols('Index~C(Class)', data=df_data_melt).fit()
    anova_table = anova_lm(model)
    # print(anova_table)
    
    f_value = anova_table['F'].iloc[0]
    p_value = anova_table['PR(>F)'].iloc[0]
    

    if p_value<0.05:
        tukey_result = pairwise_tukeyhsd(df_data_melt['Index'], df_data_melt['Class'], alpha = 0.05)
        tukey_result = tukey_result.reject
        # print(tukey_result)
        tukey_result = tukey_result.astype(np.int32)
    else:
        tukey_result = [0,0,0]
    
    return p_value, f_value, tukey_result



def statis_analysis_group():
    
    C_good, C_middle, C_bad = data_group()

    p_value_all = np.zeros((3, 6, 5, 30, 30))
    f_value_all = np.zeros((3, 6, 5, 30, 30))
    tukey_result_all = np.zeros((3, 6, 5, 30, 30, 3))
    
    for s in range(3):
        for i in range(6):
            for j in range(5):
                for m in range(30):
                    for n in range(30):
                        data = [C_good[s,:,i,j,m,n], C_middle[s,:,i,j,m,n], 
                                                C_bad[s,:,i,j,m,n]]
                        
                        p_value, f_value, tukey_result = anova_analysis(data)
                        
                        p_value_all[s,i,j,m,n] = p_value
                        f_value_all[s,i,j,m,n] = f_value
                        tukey_result_all[s,i,j,m,n] = tukey_result
                        
    return p_value_all, f_value_all, tukey_result_all             
        

def statis_analysis_stage():
    
    C_good, C_middle, C_bad = data_group()

    p_value_all = np.zeros((3, 6, 5, 30, 30))
    f_value_all = np.zeros((3, 6, 5, 30, 30))
    tukey_result_all = np.zeros((3, 6, 5, 30, 30, 3))
    
    C = [C_good, C_middle, C_bad]
    for g in range(3):
        for i in range(6):
            for j in range(5):
                for m in range(30):
                    for n in range(30):
                        data = C[g][:,:,i,j,m,n]
                        p_value, f_value, tukey_result = anova_analysis(data)
                        
                        p_value_all[g,i,j,m,n] = p_value
                        f_value_all[g,i,j,m,n] = f_value
                        tukey_result_all[g,i,j,m,n] = tukey_result
                        
    return p_value_all, f_value_all, tukey_result_all             


def statis_analysis_stage_all():
    
    C_good, C_middle, C_bad = data_group()

    p_value_all = np.zeros((6, 5, 30, 30))
    f_value_all = np.zeros((6, 5, 30, 30))
    tukey_result_all = np.zeros((6, 5, 30, 30, 3))
    
    C = np.concatenate((C_good, C_middle, C_bad), axis=1)

    for i in range(6):
        for j in range(5):
            for m in range(30):
                for n in range(30):
                    data = C[:,:,i,j,m,n]
                    p_value, f_value, tukey_result = anova_analysis(data)
                    
                    p_value_all[i,j,m,n] = p_value
                    f_value_all[i,j,m,n] = f_value
                    tukey_result_all[i,j,m,n] = tukey_result
                        
    return p_value_all, f_value_all, tukey_result_all    




# p_value_stage, f_value_stage, tukey_result_stage = statis_analysis_stage_all()

# with open('./results/statis_abalysis/stage_all/p_value_stage_all.pkl', 'wb') as f:
#     pickle.dump(p_value_stage, f)

# with open('./results/statis_abalysis/stage_all/f_value_stage_all.pkl', 'wb') as f:
#     pickle.dump(f_value_stage, f)

# with open('./results/statis_abalysis/stage_all/tukey_result_stage_all.pkl', 'wb') as f:
#     pickle.dump(tukey_result_stage, f)









#%%% plot  Differences between groups at different stages

with open('./results/statis_abalysis/group/p_value_all.pkl', 'rb') as file_obj:
    p_value_all = pickle.load(file_obj)

p_value_all[np.where(p_value_all>0.05)] = np.NAN


stage = ['Pre', 'Exp', 'Post'] 
fix_values = [0.1, 0.2, 0.3, 0.4, 0.5]  
figure_name = ['CCorr', 'Coherence', 'PLI', 'WPLI', 'Power Corr', 'PLV']

include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
               'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
               'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']


for s, p_value_stage in enumerate(p_value_all):

    for i, p_value_index in enumerate(p_value_stage):
        plt.figure(figsize=(6, 5))
        for j, p_value_one in enumerate(p_value_index):
            p_value_one[np.where(p_value_one<=0.05)] = fix_values[j] 
            sns.heatmap(p_value_one, vmin=0, vmax=1, cmap="Set1", cbar=False, center=0.5,
                        xticklabels=include_ch, yticklabels=include_ch, alpha=0.7)
            
        save_dir = './results/figure/significant difference/' + stage[s] \
                                        + '/' +figure_name[i] +'.png'
        plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
    



#%%% plot  The differences between stages in each of the three groups:

with open('./results/statis_abalysis/stage/p_value_stage.pkl', 'rb') as file_obj:
    p_value_stage = pickle.load(file_obj)

p_value_stage[np.where(p_value_stage>0.05)] = np.NAN


group = ['good', 'middle', 'bad'] 
fix_values = [0.1, 0.2, 0.3, 0.4, 0.5] 
figure_name = ['CCorr', 'Coherence', 'PLI', 'WPLI', 'Power Corr', 'PLV']

include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
               'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
               'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']


for g, p_value_group in enumerate(p_value_stage):
 
    for i, p_value_index in enumerate(p_value_group):
        plt.figure(figsize=(11, 9))
        for j, p_value_one in enumerate(p_value_index):
            p_value_one[np.where(p_value_one<=0.05)] = fix_values[j] 
            sns.heatmap(p_value_one, vmin=0, vmax=1, cmap="Set1", cbar=False, center=0.5,
                        xticklabels=include_ch, yticklabels=include_ch, alpha=0.7)
            
        save_dir = './results/figure/significant difference/' + group[g] \
                                        + '/' +figure_name[i] +'.png'
        plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
    



#%%% plot  pre-exp  exp-post 



from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle


def plot_figure(num):
    """
    num=2 pre-exp
    num=1 exp-post
    """
    
    with open('./results/statis_abalysis/stage_all/p_value_stage_all.pkl', 'rb') as file_obj:
        p_value_stage_all = pickle.load(file_obj)
    
    with open('./results/statis_abalysis/stage_all/tukey_result_stage_all.pkl', 'rb') as file_obj:
        tukey_result_stage_all = pickle.load(file_obj)    
    
    mask = tukey_result_stage_all[..., num] == 1  
    indices = np.where(mask)  

    p_value_selected = np.full_like(p_value_stage_all, np.nan, dtype=np.float64)
    p_value_selected[indices] = p_value_stage_all[indices]
    
    
    fix_values = [0.1, 0.2, 0.3, 0.4, 0.5] 
    figure_name = ['CCorr', 'Coherence', 'PLI', 'WPLI', 'Power Corr', 'PLV']

    include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
                   'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
                   'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']

    
    if num==2:
        path = './results/figure/significant difference/pre-exp'
    else:
        path = './results/figure/significant difference/exp-post'
    
    

    for i, p_value_index in enumerate(p_value_selected):
        plt.figure(figsize=(6, 5))
        for j, p_value_one in enumerate(p_value_index):
            p_value_one[np.where(p_value_one<=0.05)] = fix_values[j]
            sns.heatmap(p_value_one, vmin=0, vmax=1, cmap="Set1", cbar=False, center=0.5,
                        xticklabels=include_ch, yticklabels=include_ch, alpha=0.7)
            
        save_dir = path + '/' +figure_name[i] +'.png'
        plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
    

plot_figure(2)  
plot_figure(1)  









#%%% plot 96 teams pre-exe-post

with open('./results/statis_abalysis/stage_all/p_value_stage_all.pkl', 'rb') as file_obj:
    p_value_stage_all = pickle.load(file_obj)

with open('./results/statis_abalysis/stage_all/tukey_result_stage_all.pkl', 'rb') as file_obj:
    tukey_result_stage_all = pickle.load(file_obj)    


target = np.array([1,1,1])
mask = (tukey_result_stage_all == target).all(axis=-1)  
indices = np.where(mask)


p_value_selected = np.full_like(p_value_stage_all, np.nan, dtype=np.float64)
p_value_selected[indices] = p_value_stage_all[indices]


fix_values = [0.1, 0.2, 0.3, 0.4, 0.5]
figure_name = ['CCorr', 'Coherence', 'PLI', 'WPLI', 'Power Corr', 'PLV']

include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
               'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
               'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']



path = './results/figure/significant difference/all-stage'


for i, p_value_index in enumerate(p_value_selected):
    plt.figure(figsize=(6, 5))
    for j, p_value_one in enumerate(p_value_index):
        p_value_one[np.where(p_value_one<=0.05)] = fix_values[j] 
        sns.heatmap(p_value_one, vmin=0, vmax=1, cmap="Set1", cbar=False, center=0.5,
                    xticklabels=include_ch, yticklabels=include_ch, alpha=0.7)
        
    save_dir = path + '/' +figure_name[i] +'.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()







#%% Normal distribution histogram of drumming time difference


config = {
    "font.family":'Times New Roman', 
    "font.size": 18,
    "savefig.dpi": 300 
}
plt.rcParams.update(config)  



with open('./results/tum_time_diff_mean.pkl', 'rb') as file_obj:
     tum_time_diff = pickle.load(file_obj)
 
data = tum_time_diff.reshape((-1))

plt.figure(figsize=(5,3.5))
plt.hist(data, bins=50, density=True, alpha=0.6, color='#1F77B4', edgecolor='black')
plt.xlabel('Mean Time Difference')
plt.ylabel('Probability Density')
plt.title('Normalized Histogram')
plt.subplots_adjust(bottom=0.2)  
plt.show()








#%% Graph theory index


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle


def split_group(name):
    """
    return: row 0: good     row 1: middle    row 2: bad
    """
    
    with open(f'./results/{name}.pkl', 'rb') as file_obj:
        index_all = pickle.load(file_obj)
        
    index_all = np.transpose(index_all, (0, 2, 3, 1, 4)) 

    with open('./results/tum_time_diff_mean.pkl', 'rb') as file_obj:
        tum_time_diff = pickle.load(file_obj)
        
    tum_time_diff = np.mean(tum_time_diff, axis=1)  
    
    good_p = np.where(tum_time_diff<=0.4)[0] #  T<0.4s
    middle_p = np.where((tum_time_diff>0.4) & (tum_time_diff<0.55))[0] # 0.4s<T<0.55
    bad_p = np.where(tum_time_diff>=0.55)[0]  # T>0.55s
    
    good_group = index_all[:,:,:,good_p,:]
    middle_group = index_all[:,:,:,middle_p,:]
    bad_group = index_all[:,:,:,bad_p,:]

    return good_group, middle_group, bad_group



def anova_analysis(data):
    
    df_data = pd.DataFrame(data).T
    df_data.columns = ['G', 'M', 'B'] 
    df_data_melt = df_data.melt()
    df_data_melt = df_data_melt.dropna() 
    df_data_melt.columns = ['Class','Index']
    
    model = ols('Index~C(Class)',data=df_data_melt).fit()
    anova_table = anova_lm(model)
    # print(anova_table)
    
    f_value = anova_table['F'].iloc[0]
    p_value = anova_table['PR(>F)'].iloc[0]
    

    if p_value<0.05:
        tukey_result = pairwise_tukeyhsd(df_data_melt['Index'], df_data_melt['Class'], alpha = 0.05)
        tukey_result = tukey_result.reject
        # print(tukey_result)
        tukey_result = tukey_result.astype(np.int32)
    else:
        tukey_result = [0,0,0]
    
    return p_value, f_value, tukey_result



def statis_analysis_group_GT(name):
    """Statistical analysis of each group at different stages """
    
    good_group, middle_group, bad_group =  split_group(name)


    p_value_all = np.zeros((3, 6, 5, 4))
    f_value_all = np.zeros((3, 6, 5, 4))
    tukey_result_all = np.zeros((3, 6, 5, 4, 3))
    
    for s in range(3):
        for i in range(6):
            for j in range(5):
                for m in range(4):
                    data = [good_group[s,i,j,:,m], middle_group[s,i,j,:,m], 
                                            bad_group[s,i,j,:,m]]
                    
                    p_value, f_value, tukey_result = anova_analysis(data)
                    
                    p_value_all[s,i,j,m] = p_value
                    f_value_all[s,i,j,m] = f_value
                    tukey_result_all[s,i,j,m] = tukey_result
                        
    return p_value_all, f_value_all, tukey_result_all     



def statis_analysis_stage_GT(name):

    
    good_group, middle_group, bad_group =  split_group(name)

    indices = np.concatenate((good_group, middle_group, bad_group), axis=3)

    p_value_all = np.zeros((6, 5, 4))
    f_value_all = np.zeros((6, 5, 4))
    tukey_result_all = np.zeros((6, 5, 4, 3))
    
    for i in range(6):
        for j in range(5):
            for m in range(4):
                data = indices[:,i,j,:,m]
                
                p_value, f_value, tukey_result = anova_analysis(data)
                
                p_value_all[i,j,m] = p_value
                f_value_all[i,j,m] = f_value
                tukey_result_all[i,j,m] = tukey_result
                        
    return p_value_all, f_value_all, tukey_result_all     










# name = 'index_all'
# p_value_all, f_value_all, tukey_result_all = statis_analysis_group_GT(name)


# a = np.where(p_value_all<0.05)



# target = np.array([1,1,1])
# mask = (tukey_result_all == target).all(axis=-1)  
# indices = np.where(mask)




#%%% plot Analysis of differences at different stages


name = 'index_all_95'
p_value_all, _, tukey_result_all = statis_analysis_group_GT(name)

num = np.argwhere(p_value_all<0.05)  
i0 = num[:, 0]
i1 = num[:, 1]
i2 = num[:, 2]
i3 = num[:, 3]

mask = np.any(tukey_result_all[i0, i1, i2, i3] == 1, axis=1)
i0, i1, i2, i3 = i0[mask], i1[mask], i2[mask], i3[mask]

good_group, middle_group, bad_group =  split_group(name)

good_group = np.transpose(good_group, (0,1,2,4,3))
good_group = good_group[i0,i1,i2,i3,:]

middle_group = np.transpose(middle_group, (0,1,2,4,3))
middle_group = middle_group[i0,i1,i2,i3,:]

bad_group = np.transpose(bad_group, (0,1,2,4,3))
bad_group = bad_group[i0,i1,i2,i3,:]


stage = ['Pre', 'Exp', 'Post']
index1 = ['CCorr', 'Coh', 'PLI', 'WPLI', 'PowerCorr', 'PLV']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] 
index2 = ['GE', 'Sigma', 'LC', 'CC']

name = []
for i in range(len(i0)):
    name_ = stage[i0[i]] + '_' + index1[i1[i]] + '_' + bands[i2[i]] + '_' + index2[i3[i]] 
    name.append(name_)




df_good = pd.DataFrame(good_group, index=name)
df_mid = pd.DataFrame(middle_group, index=name)
df_bad = pd.DataFrame(bad_group, index=name)


df_good.to_excel("./results/statis_abalysis/graph_theory/good_group.xlsx", index=True)
df_mid.to_excel("./results/statis_abalysis/graph_theory/middle_group.xlsx", index=True)
df_bad.to_excel("./results/statis_abalysis/graph_theory/bad_group.xlsx", index=True)


#%%% plot 96 teams  pre-exp  exp-post


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle


def plot_figure(num):
    """
    num=2 pre-exp
    num=1 exp-post
    """
    
    name = 'index_all_95'
    p_value_all, _, tukey_result_all = statis_analysis_stage_GT(name)

    
    mask = tukey_result_all[..., num] == 1 
    indices = np.where(mask) 

    p_value_selected = np.full_like(p_value_all, np.nan, dtype=np.float64)
    p_value_selected[indices] = p_value_all[indices]
    
    
    
    fix_values = [0.1, 0.2, 0.3, 0.4] 
    yticklabels = ['CCorr', 'Coh', 'PLI', 'WPLI', 'Pow_Corr', 'PLV']

    xticklabels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']  

    
    if num==2:
        path = './results/figure/significant difference/pre-exp'
    else:
        path = './results/figure/significant difference/exp-post'
    
    
    plt.figure(figsize=(4, 3.5))
    for i in range(4):
        p_value_one = p_value_selected[:, :, i]
        p_value_one[np.where(p_value_one<=0.05)] = fix_values[i] 
        ax = sns.heatmap(p_value_one, vmin=0, vmax=1, cmap="Set1", cbar=False, center=0.5,
                    xticklabels=xticklabels, yticklabels=yticklabels, alpha=0.7)
    

    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color('black')   
        
    save_dir = path + '/GT.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()


plot_figure(2)  
plot_figure(1)  



#%%% plot 


name = 'index_all_95'
p_value_all, _, tukey_result_all = statis_analysis_stage_GT(name)


target = np.array([1,1,1])
mask = (tukey_result_all == target).all(axis=-1)  
indices = np.where(mask)


p_value_selected = np.full_like(p_value_all, np.nan, dtype=np.float64)
p_value_selected[indices] = p_value_all[indices]

fix_values = [0.1, 0.2, 0.3, 0.4] 
yticklabels = ['CCorr', 'Coh', 'PLI', 'WPLI', 'Pow_Corr', 'PLV']

xticklabels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] 

path = './results/figure/significant difference/all-stage'



plt.figure(figsize=(4, 3.5))
for i in range(4):
    p_value_one = p_value_selected[:, :, i]
    p_value_one[np.where(p_value_one<=0.05)] = fix_values[i] 
    ax = sns.heatmap(p_value_one, vmin=0, vmax=1, cmap="Set1", cbar=False, center=0.5,
                xticklabels=xticklabels, yticklabels=yticklabels, alpha=0.7)


for side in ['top', 'bottom', 'left', 'right']:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(0.5)
    ax.spines[side].set_color('black')
    
save_dir = path + '/GT.png'
plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()



































