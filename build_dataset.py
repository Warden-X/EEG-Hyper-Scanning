# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:14:55 2024

@author: Lenovo
"""


from matplotlib import pyplot as plt
import numpy as np
import h5py
import pickle

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from feat_selection_methods import feat_selection_F, feat_selection_RF, feat_selection_RFECV, feat_selection_SFS

from feature_names import feature_names




def data_import_mean():
    """
    C_all_mean   

    return: row 0: good     row 1: middle    row 2: bad
    """
    
    with open('./results/C_all_mean.pkl', 'rb') as file_obj:
        C_all_mean = pickle.load(file_obj)
    
    C_all_mean = np.swapaxes(C_all_mean, 0, 1)

    with open('./results/tum_time_diff_mean.pkl', 'rb') as file_obj:
        tum_time_diff = pickle.load(file_obj)
    
    # group
    good_p = np.array(np.where(tum_time_diff<0.4)).T # 击鼓时间差T小于0.4s T<0.4s
    middle_p = np.array(np.where((tum_time_diff>0.4) & (tum_time_diff<0.55))).T # 0.4s<T<0.55
    bad_p = np.array(np.where(tum_time_diff>0.55)).T  # T>0.55s
    
    C_good = [] # Good
    C_middle = [] # Middle
    C_bad = [] # bad
    
    for i in range(len(good_p)):
        index1 = good_p[i][0]
        index2 = good_p[i][1]
        C_one = C_all_mean[:, index1, index2]
        C_good.append(C_one)
        
    C_good = np.array(C_good)
    C_good = np.swapaxes(C_good, 0, 1) 
    num_good = np.size(C_good,1)
    C_good = C_good.reshape((3,num_good,-1))
    
    
    for i in range(len(middle_p)):
        index1 = middle_p[i][0]
        index2 = middle_p[i][1]
        C_one = C_all_mean[:, index1, index2]
        C_middle.append(C_one)
        
    C_middle = np.array(C_middle)
    C_middle = np.swapaxes(C_middle, 0, 1) 
    num_middle = np.size(C_middle,1)
    C_middle = C_middle.reshape((3,num_middle,-1))
    
    
    for i in range(len(bad_p)):
        index1 = bad_p[i][0]
        index2 = bad_p[i][1]
        C_one = C_all_mean[:, index1, index2]
        C_bad.append(C_one)
        
    C_bad = np.array(C_bad)
    C_bad = np.swapaxes(C_bad, 0, 1)
    num_bad = np.size(C_bad,1)
    C_bad = C_bad.reshape((3,num_bad,-1))
    
    
    dataset = np.concatenate((C_good, C_middle, C_bad), axis=1)
    label = np.concatenate((np.ones(num_good)+1, np.ones(num_middle), np.zeros(num_bad)))  # 标签
  
    return dataset, label



def feat_selection_mean():
    """ stages = ['pre', 'exp', 'post'] """
    
    dataset, y = data_import_mean()
    stages = ['pre', 'exp', 'post']
    
    dataset_mean = {}
    feat_index_mean = {}
    feat_names_mean = {}
    
    for n in range(len(dataset)):
        
        X = dataset[n]
        X, y, feat_index = feat_selection_F(X, y, 200) # feature selection
        feat_names = feature_names() # feature names
        feat_names = [feat_names[i] for i in feat_index] 
        
        # split dataset
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
        
        # second feature selection          
        X_train, feat_index2 = feat_selection_SFS(X_train, y_train, direction='forward') 

        feat_index = [feat_index[i] for i in feat_index2]
        feat_names = [feat_names[i] for i in feat_index2] 
        
        X_test = X_test[:,feat_index2] 
        
        dataset_one = {'X_train':X_train, 'X_test':X_test,   
                       'y_train':y_train, 'y_test':y_test}
        
        dataset_mean[f'{stages[n]}'] = dataset_one
        feat_index_mean[f'{stages[n]}'] = feat_index
        feat_names_mean[f'{stages[n]}'] = feat_names
        
    with open('./results/results/mean/dataset_mean.pkl', 'wb') as f:
        pickle.dump(dataset_mean, f)
        
    with open('./results/results/mean/feat_index_mean.pkl', 'wb') as f:
        pickle.dump(feat_index_mean, f)
        
    with open('./results/results/mean/feat_names_mean.pkl', 'wb') as f:
        pickle.dump(feat_names_mean, f)

    
    return dataset_mean




#%% main

if __name__ == '__main__':
    
    dataset, label = data_import_mean()
    
    dataset_mean = feat_selection_mean()

    











