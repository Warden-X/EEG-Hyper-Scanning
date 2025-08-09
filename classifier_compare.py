# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:23:47 2022

@author: comeo
"""

import numpy as np
import pickle

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def SVM_classifier(X_train, y_train, C, gamma):
    
    clf = svm.SVC(C=C, gamma=gamma) 
    score = []
    # 10-time 10-fold CV
    for i in range(10):
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=20)
        score.append(np.mean(scores))
    score = np.mean(score)
    print(f'{C} {gamma}: {score}')
    
    return score


def SVM_results(X_train, y_train):
    
    results = {}
    C_para = np.arange(100,1000, 50) 
    gamma_para = np.linspace(0.01, 0.5, 50)
    
    acc = []
    for m in range(len(gamma_para)):  # gamma
        gamma = gamma_para[m]
        acc_ = []
        for n in range(len(C_para)):  # C
            C = C_para[n]  
            score = SVM_classifier(X_train, y_train, C, gamma)
            acc_.append(score)
            
        acc.append(acc_)
    
    acc =  np.array(acc)
    acc_max = np.max(acc)
    index = np.array(np.where(acc == acc_max))[:,0]
    
    gamma_ = gamma_para[index[0]]
    C_ = C_para[index[1]]
    
    results['acc'] = acc
    results['acc_max'] = acc_max
    results['index_optimal'] = [C_, gamma_]
        
    return results


def RF_classifier(X_train, y_train, n, m):
    
    clf = RandomForestClassifier(n_estimators=n, max_depth=m)
    
    score = []
    # 10-10 CV
    for i in range(10):
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=20)
        score.append(np.mean(scores))
    score = np.mean(score)
    print(f'{n} {m}: {score}')
    
    return score


def RF_results(X_train, y_train):
    
    results = {}
    para_n = np.arange(50,250,10) 
    para_m = np.arange(5,30,2)  
    
    acc = []
    for n in para_n:  # n
        acc_ = []
        for m in para_m:  # m
            score = RF_classifier(X_train, y_train, n, m)
            acc_.append(score)
            
        acc.append(acc_)
        
    acc =  np.array(acc)
    acc_max = np.max(acc)
    index = np.array(np.where(acc == acc_max))[:,0]
    
    n_ = para_n[index[0]]
    m_ = para_m[index[1]]
    
    results['acc'] = acc
    results['acc_max'] = acc_max
    results['index_optimal'] = [n_, m_]
        
    return results


def KNN_classifier(X_train, y_train, n_neighbors, leaf_size):

    clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=n_neighbors, leaf_size=leaf_size) 
    score = []

    for i in range(10):
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=20)
        score.append(np.mean(scores))
    score = np.mean(score)
    print(f'{n_neighbors} {leaf_size}: {score}')
    return score



def KNN_results(X_train, y_train):
    
    results = {}
    n_neighbors_para = np.arange(5, 100, 5)  
    leaf_size_para = np.arange(10, 300, 10)
    
    acc = []
    for m in range(len(n_neighbors_para)):  
        n_neighbors = n_neighbors_para[m]
        acc_ = []
        for n in range(len(leaf_size_para)):  
            leaf_size = leaf_size_para[n]  
            score = KNN_classifier(X_train, y_train, n_neighbors, leaf_size)
            acc_.append(score)
            
        acc.append(acc_)
        
    acc =  np.array(acc)
    acc_max = np.max(acc)
    index = np.array(np.where(acc == acc_max))[:,0]
    
    n_neighbors_ = n_neighbors_para[index[0]]
    leaf_size_ = leaf_size_para[index[1]]
        
    results['acc'] = acc
    results['acc_max'] = acc_max
    results['index_optimal'] = [n_neighbors_, leaf_size_]
        
    return results


def LDA_classifier(X_train, y_train, shrinkage):
        
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage) 
    score = []

    for i in range(10):
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=20)
        score.append(np.mean(scores))
    score = np.mean(score)
    print(f'{shrinkage}: {score}')
    return score


def LDA_results(X_train, y_train):
    
    results = {}
    shrinkage_para =np.linspace(0.01, 1, 100) 
    LDA_acc = []
    for shrinkage in shrinkage_para:
        LDA_acc_ = LDA_classifier(X_train, y_train, shrinkage)
        LDA_acc.append(LDA_acc_)
    
    acc =  np.array(LDA_acc)
    acc_max = np.max(acc)
    index = np.array(np.where(acc == acc_max))[0]
    
    shrinkage_ = shrinkage_para[index[0]]
        
    results['acc'] = acc
    results['acc_max'] = acc_max
    results['index_optimal'] = shrinkage_
        
    return results
    


def NB_classifier(X_train, y_train, var_smoothing):

    clf = GaussianNB(var_smoothing=var_smoothing) 
    score = []

    for i in range(10):
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=20)
        score.append(np.mean(scores))
    score = np.mean(score)
    print(f'{var_smoothing}: {score}')
    
    return score


def NB_results(X_train, y_train):
    
    results = {}
    var_smoothing_para = [ 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5]
    NB_acc = []
    for var_smoothing in var_smoothing_para:
        NB_acc_= NB_classifier(X_train, y_train, var_smoothing)
        NB_acc.append(NB_acc_)
    
    acc =  np.array(NB_acc)
    acc_max = np.max(acc)
    index = np.array(np.where(acc == acc_max))[0]
    
    var_smoothing_ = var_smoothing_para[index[0]]
        
    results['acc'] = acc
    results['acc_max'] = acc_max
    results['index_optimal'] = var_smoothing_
        
    return results


def LR_results(X_train, y_train):
    
    clf = LogisticRegression(max_iter=500)
    
    score = []

    for i in range(10):
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=20)
        score.append(np.mean(scores))
    score = np.mean(score)
    print(f'LR: {score}')

    return score


def ml_test(X_train, y_train, X_test, y_test, key_word, index):
    
    """buliding optimal model """
    
    if key_word=='SVM':
        clf = svm.SVC(C=index[0], gamma=index[1])
    elif key_word=='RF':
        clf = RandomForestClassifier(n_estimators=index[0], max_depth=index[1])
    elif key_word=='KNN':
        clf = KNeighborsClassifier(algorithm='kd_tree', 
                      n_neighbors=index[0], leaf_size=index[1])
    elif key_word=='LDA':
        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=index)
    elif key_word=='NB':
        clf = GaussianNB(var_smoothing=index)
    elif key_word=='LR':
        clf = LogisticRegression(max_iter=500)
    else:
        print('The key word is wrong!')
    

    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
        
    return score, f1






#%% main
if __name__ == '__main__':
    

    with open('./results/results/mean/dataset_mean.pkl', 'rb') as file_obj:
        dataset_mean = pickle.load(file_obj)
    
    
    # find optimal parameters
    SVM_acc, RF_acc, KNN_acc = {}, {}, {} 
    LDA_acc, NB_acc, LR_acc  = {}, {}, {}  

    for stage, dataset_stage in dataset_mean.items():
        
        X_train = dataset_stage['X_train']
        y_train = dataset_stage['y_train']

        SVM_acc[f'{stage}'] = SVM_results(X_train, y_train)
        RF_acc[f'{stage}']  = RF_results(X_train, y_train)
        KNN_acc[f'{stage}'] = KNN_results(X_train, y_train)
        LDA_acc[f'{stage}'] = LDA_results(X_train, y_train)
        NB_acc[f'{stage}']  = NB_results(X_train, y_train)
        LR_acc[f'{stage}']  = LR_results(X_train, y_train)

    results_algorithms_tuning = {}
    algorithms = ['SVM', 'RF', 'KNN', 'LDA', 'NB', 'LR']
    for i, acc in enumerate([SVM_acc, RF_acc, KNN_acc, LDA_acc, NB_acc, LR_acc]):
        results_algorithms_tuning[f'{algorithms[i]}'] = acc

    with open('./results/results/mean/results_algorithms_tuning.pkl', 'wb') as f:
        pickle.dump(results_algorithms_tuning, f)
    
    
    
    

    # The performance of optimal  model is tested on the test set
    with open('./results/results/mean/dataset_mean.pkl', 'rb') as file_obj:
        dataset_mean = pickle.load(file_obj)
        
    with open('./results/results/mean/results_algorithms_tuning.pkl', 'rb') as file_obj:
        results_algorithms_tuning = pickle.load(file_obj)
    
    
    results_algorithms_test = {}
    for key_word, acc in results_algorithms_tuning.items():
        results_algorithms_test[key_word] = {}
        for stage, rsults in acc.items():
            X_train = dataset_mean[stage]['X_train']
            y_train = dataset_mean[stage]['y_train']
            X_test = dataset_mean[stage]['X_test']
            y_test = dataset_mean[stage]['y_test']
            if key_word=='LR':
                results_test_ = ml_test(X_train, y_train, X_test, y_test, key_word, index=0)
                results_algorithms_test[key_word][stage] = results_test_
            else:
                index = rsults['index_optimal']
                results_test_ = ml_test(X_train, y_train, X_test, y_test, key_word, index)
                results_algorithms_test[key_word][stage] = results_test_
            
            
    with open('./results/results/mean/results_algorithms_test.pkl', 'wb') as f:
        pickle.dump(results_algorithms_test, f)       
        
    
    
    
    #%%% ALL stages features

    with open('./results/results/mean/dataset_mean.pkl', 'rb') as file_obj:
        dataset_mean = pickle.load(file_obj)
        
    with open('./results/results/mean/feat_names_mean.pkl', 'rb') as file_obj:
        feat_names_mean = pickle.load(file_obj)

    feat_names = [name+'_pre' for name in feat_names_mean['pre']] + \
                  [name+'_exp' for name in feat_names_mean['exp']] + \
                  [name+'_post' for name in feat_names_mean['post']]
    
     
    with open('./results/results/mean/All_Stages/feat_names_300.pkl', 'wb') as f:
        pickle.dump(feat_names, f)
                 

    dataset_mean_all_stages = {}
    
    dataset_mean_all_stages['X_train'] = np.concatenate((dataset_mean['pre']['X_train'], 
                                  dataset_mean['exp']['X_train'], 
                                  dataset_mean['post']['X_train']), axis=1)
    dataset_mean_all_stages['y_train']  = dataset_mean['pre']['y_train']

    dataset_mean_all_stages['X_test']  = np.concatenate((dataset_mean['pre']['X_test'], 
                                  dataset_mean['exp']['X_test'], 
                                  dataset_mean['post']['X_test']), axis=1)
    dataset_mean_all_stages['y_test']  = dataset_mean['pre']['y_test']


    with open('./results/results/mean/All_Stages/dataset_mean_all_stages.pkl', 'wb') as f:
        pickle.dump(dataset_mean_all_stages, f)  



    # adjusting parameters
    SVM_acc= SVM_results(dataset_mean_all_stages['X_train'] , dataset_mean_all_stages['y_train'] )
    RF_acc= RF_results(dataset_mean_all_stages['X_train'] , dataset_mean_all_stages['y_train'] )
    KNN_acc = KNN_results(dataset_mean_all_stages['X_train'] , dataset_mean_all_stages['y_train'] )
    LDA_acc= LDA_results(dataset_mean_all_stages['X_train'] , dataset_mean_all_stages['y_train'] )
    NB_acc  = NB_results(dataset_mean_all_stages['X_train'] , dataset_mean_all_stages['y_train'] )
    LR_acc = LR_results(dataset_mean_all_stages['X_train'] , dataset_mean_all_stages['y_train'] )

    results_all_stages_tuning = {}
    algorithms = ['SVM', 'RF', 'KNN', 'LDA', 'NB', 'LR']
    for i, acc in enumerate([SVM_acc, RF_acc, KNN_acc, LDA_acc, NB_acc, LR_acc]):
        results_all_stages_tuning[f'{algorithms[i]}'] = acc

    with open('./results/results/mean/All_Stages/results_all_stages_tuning.pkl', 'wb') as f:
        pickle.dump(results_all_stages_tuning, f)

    # test on testset
    with open('./results/results/mean/All_Stages/dataset_mean_all_stages.pkl', 'rb') as f:
        dataset_mean_all_stages = pickle.load(f)  
        
    with open('./results/results/mean/All_Stages/results_all_stages_tuning.pkl', 'rb') as f:
        results_all_stages_tuning = pickle.load(f)



    results_all_stages_test = {}
    for key_word, rsults in results_all_stages_tuning.items():

        X_train = dataset_mean_all_stages['X_train']
        y_train = dataset_mean_all_stages['y_train']
        X_test = dataset_mean_all_stages['X_test']
        y_test = dataset_mean_all_stages['y_test']
        if key_word=='LR':
            results_test_ = ml_test(X_train, y_train, X_test, y_test, key_word, index=0)
            results_all_stages_test[key_word]= results_test_
        else:
            index = rsults['index_optimal']
            results_test_ = ml_test(X_train, y_train, X_test, y_test, key_word, index)
            results_all_stages_test[key_word]= results_test_
        
            
    with open('./results/results/mean/All_Stages/results_all_stages_test.pkl', 'wb') as f:
        pickle.dump(results_all_stages_test, f)       



    index = results_all_stages_tuning['SVM']['index_optimal']
    clf_SVM = svm.SVC(C=index[0], gamma=index[1])
    clf_SVM = clf_SVM.fit(X_train, y_train)
    score = clf_SVM.score(X_test, y_test)
    
    with open('./results/results/mean/All_Stages/clf_SVM.pkl', 'wb') as f:
        pickle.dump(clf_SVM, f)
    
    
    index = results_all_stages_tuning['LDA']['index_optimal']
    clf_LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage=index)
    clf_LDA = clf_LDA.fit(X_train, y_train)
    score = clf_LDA.score(X_test, y_test)

    with open('./results/results/mean/All_Stages/clf_LDA.pkl', 'wb') as f:
        pickle.dump(clf_LDA, f)       


     
    
















