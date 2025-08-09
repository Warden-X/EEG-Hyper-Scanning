# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 03:12:08 2025

@author: Chen Min
"""


import numpy as np
import shap
import pandas as pd
import pickle
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import classification_report, f1_score
from feature_names import feature_names



config = {
    "savefig.dpi": 300,  
    'font.size': 18,              
    'font.family': 'Times New Roman',  
    'axes.titlesize': 18,          
    'axes.labelsize': 18,       
    'xtick.labelsize': 16,       
    'ytick.labelsize': 16,         
    'legend.fontsize': 16,        
}
plt.rcParams.update(config)  



#%% three stages


def bar_color(i):
    colors = ['#f59311', '#63b2ee', '#1F77B4']
    return colors[i]


with open('./results/results/mean/feat_index_mean.pkl', 'rb') as f:
    feat_index_mean =  pickle.load(f)

with open('./results/results/mean/dataset_mean.pkl', 'rb') as file_obj:
    dataset_mean = pickle.load(file_obj)

with open('./results/results/mean/results_algorithms_tuning.pkl', 'rb') as file_obj:
    results_algorithms_tuning = pickle.load(file_obj)



feat_names_all = feature_names() 


stages = ['pre', 'exp', 'post']


for stage in stages:
    X_train = dataset_mean[stage]['X_train']
    y_train = dataset_mean[stage]['y_train']
    X_test = dataset_mean[stage]['X_test']
    y_test = dataset_mean[stage]['y_test']
    
    feat_index = feat_index_mean[stage]
    feat_names = [feat_names_all[i] for i in feat_index] 
    
    X_train = pd.DataFrame(data = X_train, columns=feat_names)
    X_test = pd.DataFrame(data = X_test, columns=feat_names)
    
    
    index = results_algorithms_tuning['SVM'][stage]['index_optimal']
    clf_SVM = svm.SVC(C=index[0], gamma=index[1], probability=True)
    clf_SVM = clf_SVM.fit(X_train, y_train)

    explainer = shap.KernelExplainer(clf_SVM.predict_proba, shap.kmeans(X_train, 5), link="logit")
    shap_values = explainer.shap_values(X_test)
    

    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=['Bad','Middle','Good'],
                      max_display=20, show=False, class_inds=[2,1,0], color=bar_color)
    plt.xlabel("Mean SHAP Value")
    
    save_dir = f'./results/figure/SHAP/{stage}.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    







#%% ALL

with open('./results/results/mean/All_Stages/dataset_mean_all_stages.pkl', 'rb') as file_obj:
    dataset_mean_all_stages = pickle.load(file_obj)

X_train = dataset_mean_all_stages['X_train']
y_train = dataset_mean_all_stages['y_train']
X_test = dataset_mean_all_stages['X_test']
y_test = dataset_mean_all_stages['y_test']


with open('./results/results/mean/feat_index_mean.pkl', 'rb') as f:
    feat_index_mean =  pickle.load(f)
    

with open('./results/results/mean/feat_names_mean.pkl', 'rb') as file_obj:
    feat_names_mean = pickle.load(file_obj)

feat_names_all = feature_names() 

stages = ['pre', 'exp', 'post']

for stage in stages:
    feat_index = feat_index_mean[stage]
    feat_names_mean[stage] = [feat_names_all[i] for i in feat_index] 




feat_names = [name+'_Pre' for name in feat_names_mean['pre']] + \
              [name+'_Exe' for name in feat_names_mean['exp']] + \
              [name+'_Rec' for name in feat_names_mean['post']]
    
    

X_train = pd.DataFrame(data = X_train, columns=feat_names)
X_test = pd.DataFrame(data = X_test, columns=feat_names)

with open('./results/results/mean/All_Stages/results_all_stages_tuning.pkl', 'rb') as file_obj:
    results_all_stages_tuning = pickle.load(file_obj)





# SVM
index = results_all_stages_tuning['SVM']['index_optimal']
clf_SVM = svm.SVC(C=index[0], gamma=index[1], probability=True)
clf_SVM = clf_SVM.fit(X_train, y_train)
# clf_SVM.score(X_test, y_test)



# SHAP
explainer = shap.KernelExplainer(clf_SVM.predict_proba, shap.kmeans(X_train, 5), link="logit")
shap_values = explainer.shap_values(X_test)



def bar_color(i):
    colors = ['#f59311', '#63b2ee', '#1F77B4']
    return colors[i]

shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=['Bad','Middle','Good'],
                  max_display=20, show=False, class_inds=[2,1,0], color=bar_color)
plt.xlabel("Mean SHAP Value")

save_dir = './results/figure/SHAP/ALL.png'
plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()





shap.summary_plot(shap_values[:,:,2], X_test, max_display=30, show=False)
plt.gcf().set_size_inches(5, 10) 
plt.gcf().set_dpi(300)  
plt.tight_layout(pad=1)  
plt.show()


save_dir = './results/figure/SHAP/good.png'
plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()



 
#%%  2D 

import hypyp
from NDFSysMNE import mneNDF
import mne


def ch_inf(file_path):
    
    ndfMneObj = mneNDF(file_path) 
    data = ndfMneObj.read2MneRaw() 
    
    include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
                   'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
                   'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
    
    data_sel = data.pick(include_ch) 
    data_sel.set_montage(montage="standard_1020")

    return data_sel


shap_sum = np.sum(np.abs(shap_values), axis=(0,2))

top20_indices = np.argsort(shap_sum)[-20:][::-1]
shap_value_20 = shap_sum[top20_indices]

feature = [feat_names[i] for i in top20_indices]

non_zero_pairs  = [feature_.split('_')[-3:-1] for feature_ in feature]

include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
              'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
              'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']

matrix = np.zeros((30, 30))
ch_index = {ch: i for i, ch in enumerate(include_ch)}

for (ch1, ch2), val in zip(non_zero_pairs, shap_value_20):
    i, j = ch_index[ch1], ch_index[ch2]
    matrix[i, j] = val

    
    

file_path = r'F:\EEG Hypersacan\data\correct\20240415143724_24.4.15下午\1'
data_sel = ch_inf(file_path)

data_sel.set_montage('biosemi32')
hypyp.viz.viz_2D_topomap_inter(data_sel, data_sel, matrix, lab=True)









