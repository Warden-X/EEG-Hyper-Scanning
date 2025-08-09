# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:14:46 2024

@author: Chen Min
"""


def feature_names():

    indexes = [ 'CCorr', 'Coh', 'PLI', 'WPLI', 'Pow_Corr', 'PLV'] # 
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] 
    include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
                   'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
                   'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
    
    names = [] 
    for num in range(6*5*30*30):
        
        Q1 = num // 4500  
        index_ = indexes[Q1] 
        
        M1 = num % 4500  
        Q2 = M1//900  
        band_ = bands[Q2]
        
        M2 = M1 % 900  
        Q3 = M2 // 30 
        Q4 = M2 % 30 
        
        name = index_ + '_' + band_ + '_' + include_ch[Q3] + '_' + include_ch[Q4]
        names.append(name)

    return names











