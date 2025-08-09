# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:01:29 2024

@author: Lenovo
"""


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import  SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, f_classif


def feat_selection_F(X, y, k):
    
    feat_ =  SelectKBest(f_classif, k=k)
    X_new =feat_.fit_transform(X, y) # 根据ANOVA F值筛选特征
    feat_index = feat_.get_support(indices=True)
    
    return X_new, y, feat_index




def feat_selection_SFS(X, y, direction='forward'):
    """特征选择 SFS —— 顺序特征选择"""
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.2)

    feat_ = SequentialFeatureSelector(clf, n_features_to_select=100, cv=10, 
                        direction=direction, n_jobs=20)
    X_SFS = feat_.fit_transform(X, y)
    feat_index = feat_.get_support(indices=True)
    
    return X_SFS, feat_index






