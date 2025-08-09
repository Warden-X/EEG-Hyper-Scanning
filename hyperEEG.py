# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:48:04 2024

@author: Lenovo
"""


from NDFSysMNE import mneNDF
import mne
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import hypyp
from spkit.eeg import ATAR
import mne_connectivity
import seaborn as sns
import h5py
import pickle
import itertools



def find_file_dir(root_dir):

    p = Path(root_dir)
    full_names = [] 
    for file_dir in p.iterdir(): 
        if (file_dir.is_dir()):
            full_names.append(str(file_dir))
    return full_names


def data_input(file_path):
    """Data import and undergo preliminary preprocessing
        1. Channel selection
        2. 0.5 to 50Hz band-pass filter
        3. Downsample to 200Hz
        4. Wavelet denoising
        5. Positioning electrode position information (10-20 system)
        6. Average reference
    """
    ndfMneObj = mneNDF(file_path) 
    data = ndfMneObj.read2MneRaw() 
    
    include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
                   'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
                   'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
    

    data_sel = data.pick(include_ch) 

    data_sel = data_sel.filter(l_freq=0.5, h_freq=50, verbose=False) 
    data_sel.resample(sfreq=200)  

    
    info = mne.create_info(ch_names=include_ch, sfreq=200, ch_types='eeg')

    X = data_sel.get_data().T * 10**6 
    XR = ATAR(X, wv='db3', winsize=200, beta=0.3, IPR=[25,75], OptMode='soft')
    data_denosing = mne.io.RawArray(XR.T*10**-6, info)
   
    data_denosing.set_montage(montage="standard_1020")

    
    data_denosing.set_eeg_reference(ref_channels='average') 
    
    return data_denosing


def effective_tum(everygroup_dir):
    """Effective drumming time points within the group"""
    
    time_group = [] 
    for everyone_dir in everygroup_dir[0:4]:
    
        datatum, times =  mneNDF(everyone_dir).read2MneRaw().filter(l_freq=2,
                h_freq=50, verbose=False).pick('GSR').resample(sfreq=200)[:] 

        datatum = datatum[0] 
        sample_num = 50
        for i in range(len(datatum)//sample_num):
            datatum[(i*sample_num):(i+1)*sample_num] = datatum[(i*sample_num):\
                (i+1)*sample_num] - np.mean(datatum[(i*sample_num):(i+1)*sample_num] )
        datatum = np.abs(datatum) 

        flag_tum = np.where(datatum>0.01) 

        first_point = np.where(np.diff(flag_tum[0])>1000)[0]+1 
        

        event_times = np.append(flag_tum[0][0], flag_tum[0][first_point]) 
        
        missing_point = [] 
        missing_values = [] 
        diff_time = np.diff(event_times) 
        for index, diff_time_ in enumerate(diff_time):
            value_insert=[]
            for i in range(1, 11): 
                if diff_time_ > (i*2000+1000):
                    value_insert = [0] * i
            
            if value_insert:
                missing_point.append(index+1)
                missing_values.append(np.array(value_insert))
            

        event_sup = event_times[:]
        num = 0
        for i in range(len(missing_point)):
            event_sup = np.insert(event_sup, missing_point[i]+num, missing_values[i])
            num = num + len(missing_values[i])
        
        time_group.append(event_sup)

    time_group = np.array(time_group)
    time_group = time_group[:, 0:48] 

    return time_group


def tum_standard(full_test):
    """Time difference of drumming ———— gold standard"""
    correct_group = [] 
    error_group = [] 
    tum_time_diff = [] 
    time_group_all = []
    
    for single_test in full_test:
        everygroup_dir = find_file_dir(single_test) 
        
        try:
            time_group = effective_tum(everygroup_dir)
            correct_group.append(single_test)
            c_en = True
        except (ValueError, AttributeError, IndexError) as e:
            print(f"{e}"+': '+ single_test)
            error_group.append(single_test) 
            c_en = False
            
        if c_en:
            time_diff_all = [] 
            remove_time = np.where(time_group==0)
            time_group = np.delete(time_group, remove_time[1], axis=1) 
            time_group_all.append(time_group) 
            
            for i in range(0,4):
                time_diff = (time_group[i] - time_group[0])/200  
                time_diff_all.append(time_diff)
            tum_time_diff.append(np.mean(time_diff_all, axis=1))

    return correct_group, time_group_all, tum_time_diff, error_group


def data_split(everygroup_dir, time_group):
    """
    describe: Preparation: -2.5 to -0.5   Execution: -0.5 to 1.5   Recovery: 1.5 to 3.5
    ouput: epochs_group
    """
    epochs_group_pre = [] 
    epochs_group_exp = [] 
    epochs_group_post = [] 
    
    for everyone_dir in everygroup_dir[0:4]:
        data_denosing = data_input(everyone_dir) 

        event_times = time_group[0] 
        event =  np.vstack((event_times, np.repeat(0, len(event_times)), 
                                 np.repeat(1, len(event_times)))).T
            

        epochs_pre = mne.Epochs(data_denosing, events=event, tmin=-5, tmax=5, 
                        baseline=(None,-1), preload=True, reject=dict(eeg=2e-3))\
                        .crop(tmin=-2.5, tmax=-0.5, verbose=False)
                        
        epochs_exp = mne.Epochs(data_denosing, events=event, tmin=-5, tmax=5, 
                        baseline=(None,-1), preload=True, reject=dict(eeg=2e-3))\
                        .crop(tmin=-0.5, tmax=1.5, verbose=False)
        
        epochs_post = mne.Epochs(data_denosing, events=event, tmin=-5, tmax=5, 
                        baseline=(None,-1), preload=True, reject=dict(eeg=2e-3))\
                        .crop(tmin=1.5, tmax=3.5, verbose=False)

        
        epochs_group_pre.append(epochs_pre)
        epochs_group_exp.append(epochs_exp)
        epochs_group_post.append(epochs_post)
        
    return epochs_group_pre, epochs_group_exp, epochs_group_post


def bulid_dataset(correct_group, time_group_all):
    """Solve the EEG hyperscan indicators to construct the dataset"""
    
    epochs_all_pre = [] 
    epochs_all_exp = [] 
    epochs_all_post = [] 
    
    C_all  = [] 
    error_group_index = [] 
    
    for index, single_test in enumerate(correct_group):
        everygroup_dir = find_file_dir(single_test) 
        try:
            epochs_group_pre, epochs_group_exp, epochs_group_post = \
                    data_split(everygroup_dir, time_group_all[index]) 
            epochs_all_pre.append(epochs_group_pre)        
            epochs_all_exp.append(epochs_group_exp)  
            epochs_all_post.append(epochs_group_post)  
            
        except (ValueError, AttributeError, IndexError, RuntimeError) as e:
            print(f"{e}"+': '+ single_test)
            error_group_index.append(index)
            continue  
    
        freq_bands = {'delta':[1, 4], 'theta':[4,8], 'alpha':[8,13], 
                      'beta':[13,30], 'gamma':[30,49]}
        modes = [ 'ccorr', 'coh', 'pli', 'wpli', 'pow_corr', 'plv'] 
        
        n_ch = 30 # EEG channels
        
        C_all_ = []  
        for epochs_group in [epochs_group_pre, epochs_group_exp, epochs_group_post]:
            C_group = []
            members = [1, 2, 3]  # 成员编号

            for i, j in itertools.combinations(members, 2):
                
                selected_array  = [epochs_group[m] for m in [i,j]]
                data = np.array(selected_array)  
                
                complex_signal = hypyp.analyses.compute_freq_bands(data=data, 
                                     sampling_rate=200, freq_bands=freq_bands)
                C_mode = []
                for mode in modes:
                    result = hypyp.analyses.compute_sync(complex_signal= \
                                complex_signal, mode=mode, epochs_average=False)                    
                        
                    C_mode.append(result[:, :, 0:n_ch, n_ch:2*n_ch])  
                C_group.append(C_mode)
        
            C_all_.append(C_group)
            
        C_all.append(C_all_)


    return epochs_all_pre, epochs_all_exp, epochs_all_post, C_all, error_group_index






#%% main

if __name__ == '__main__':
    root_dir = r'F:\EEG Hypersacan\data\correct'  
    full_test = find_file_dir(root_dir) 
    correct_group, time_group_correct, tum_time_diff_correct, error_group = tum_standard(full_test)
    
    
    
    
    with open('./results/correct/time_group_correct.pkl', 'rb') as file_obj:
        time_group_correct = pickle.load(file_obj)
    correct_group = full_test
    
    epochs_all_pre, epochs_all_exp, epochs_all_post, C_all, \
          error_group_index = bulid_dataset(correct_group, time_group_correct)



    with open('./results/correct/epochs_all_pre_correct.pkl', 'wb') as f:
        pickle.dump(epochs_all_pre, f)
        
    with open('./results/correct/epochs_all_exp_correct.pkl', 'wb') as f:
        pickle.dump(epochs_all_exp, f)
        
    with open('./results/correct/epochs_all_post_correct.pkl', 'wb') as f:
        pickle.dump(epochs_all_post, f)

    with open('./results/correct/C_all_correct.pkl', 'wb') as f:
        pickle.dump(C_all, f)





    
    
    
    #%% remaining_data
    
    root_dir = r'F:\EEG Hypersacan\data\error'  
    full_test = find_file_dir(root_dir)
    
    
    time_all_error = []
    for single_test in full_test:
        everygroup_dir = find_file_dir(single_test) 
    
        time_group = [] 
        len_event = [] 
        for everyone_dir in everygroup_dir[0:4]:
        
            datatum, times =  mneNDF(everyone_dir).read2MneRaw().filter(l_freq=2,
                    h_freq=50, verbose=False).pick('GSR').resample(sfreq=200)[:] 

            datatum = datatum[0] 
            sample_num = 50
            for i in range(len(datatum)//sample_num):
                datatum[(i*sample_num):(i+1)*sample_num] = datatum[(i*sample_num):\
                    (i+1)*sample_num] - np.mean(datatum[(i*sample_num):(i+1)*sample_num] )
            datatum = np.abs(datatum) 
            

            flag_tum = np.where(datatum>0.01) 

            first_point = np.where(np.diff(flag_tum[0])>1000)[0]+1 

            event_times = np.append(flag_tum[0][0], flag_tum[0][first_point]) 
            

            missing_point = [] 
            missing_values = []
            diff_time = np.diff(event_times)
            for index, diff_time_ in enumerate(diff_time):
                value_insert=[]
                for i in range(1, 11): 
                    if diff_time_ > (i*2000+1000):
                        value_insert = [0] * i 
                
                if value_insert:
                    missing_point.append(index+1)
                    missing_values.append(np.array(value_insert))
                

            event_sup = event_times[:]
            num = 0
            for i in range(len(missing_point)):
                event_sup = np.insert(event_sup, missing_point[i]+num, missing_values[i])
                num = num + len(missing_values[i])
            
    
            len_event.append(len(event_sup))  
            time_group.append(event_sup)
        

        if np.min(len_event)<48:
            len_event_unify = np.min(len_event)    
        else:
            len_event_unify = 48
        
        for i in range(4):
            if len(time_group[i]) != len_event_unify:
                time_group[i] = time_group[i][-len_event_unify::]
        
        time_all_error.append(time_group)
        
    
    
    time_group = np.array(time_group) 
    time_group = time_group[:, 0:48] 

    
    
    with open('./results/time_all_error.pkl', 'wb') as f:
        pickle.dump(time_all_error, f)
    
    
    
    
    
    #%% main2

    
    with open('./results/error/time_all_error_raw.pkl', 'rb') as file_obj:
        time_all_error_raw = pickle.load(file_obj)

    time_all_error = [] 
    tum_time_diff_error = []
    
    for time_group in time_all_error_raw:
        time_diff_all = [] 
        
        remove_time = np.where(time_group==0)
        time_group = np.delete(time_group, remove_time[1], axis=1) 
        time_all_error.append(time_group)
 
        for i in range(0,4):
            time_diff = (time_group[i] - time_group[0])/200   
            time_diff_all.append(time_diff)
        tum_time_diff_error.append(np.mean(time_diff_all, axis=1))
    
    
    tum_time_diff_error = np.array(tum_time_diff_error)
    

    
    
    
    
    
    root_dir = r'F:\EEG Hypersacan\data\error' 
    full_test = find_file_dir(root_dir) 
    
    with open('./results/error/time_all_error.pkl', 'rb') as file_obj:
        time_all_error = pickle.load(file_obj)
        
    epochs_all_pre, epochs_all_exp, epochs_all_post, C_all, \
          error_group_index  = bulid_dataset(full_test, time_all_error)
    
    
    
    
    with open('./results/error/epochs_all_pre_error.pkl', 'wb') as f:
        pickle.dump(epochs_all_pre, f)
        
    with open('./results/error/epochs_all_exp_error.pkl', 'wb') as f:
        pickle.dump(epochs_all_exp, f)
        
    with open('./results/error/epochs_all_post_error.pkl', 'wb') as f:
        pickle.dump(epochs_all_post, f)

    with open('./results/error/C_all_error.pkl', 'wb') as f:
        pickle.dump(C_all, f)
    
    


#%% 

    import numpy as np
    import pickle

    with open('./results/correct/epochs_all_pre_correct.pkl', 'rb') as f:
        epochs_all_pre_correct = pickle.load(f)

    with open('./results/correct/epochs_all_exp_correct.pkl', 'rb') as f:
        epochs_all_exp_correct = pickle.load(f)

    with open('./results/correct/epochs_all_post_correct.pkl', 'rb') as f:
        epochs_all_post_correct = pickle.load(f)


    with open('./results/error/epochs_all_pre_error.pkl', 'rb') as f:
        epochs_all_pre_error = pickle.load(f)

    with open('./results/error/epochs_all_exp_error.pkl', 'rb') as f:
        epochs_all_exp_error = pickle.load(f)

    with open('./results/error/epochs_all_post_error.pkl', 'rb') as f:
        epochs_all_post_error = pickle.load(f)



    epochs_all_pre = epochs_all_pre_correct + epochs_all_pre_error
    epochs_all_exp = epochs_all_exp_correct + epochs_all_exp_error
    epochs_all_post = epochs_all_post_correct + epochs_all_post_error

    
    with open('./results/epochs_all_pre.pkl', 'wb') as f:
        pickle.dump(epochs_all_pre, f)
        
    with open('./results/epochs_all_exp.pkl', 'wb') as f:
        pickle.dump(epochs_all_exp, f)

    with open('./results/epochs_all_post.pkl', 'wb') as f:
        pickle.dump(epochs_all_post, f)


    #%%% 
    import numpy as np
    import pickle
    
    with open('./results/correct/time_group_correct.pkl', 'rb') as f:
        time_group_correct = pickle.load(f)
        
    with open('./results/error/time_all_error.pkl', 'rb') as f:
        time_group_error = pickle.load(f)        
    
    
    time_group = time_group_correct + time_group_error
    
    with open('./results/time_group.pkl', 'wb') as f:
        pickle.dump(time_group, f) 
    
    
    
    

    tum_time_diff = []
    for i in range(len(time_group)):
        time_diff_group = [] 
        for j in range(0,4):
            time_diff = (time_group[i][j] - time_group[i][0])/200   # 时间差
            time_diff_group.append(time_diff)

        tum_time_diff.append(np.array(time_diff_group))

    with open('./results/tum_time_diff_all.pkl', 'wb') as f:
        pickle.dump(tum_time_diff, f) 



    tum_time_diff_mean = []
    for i in range(len(tum_time_diff)):
        time_diff_mean = np.mean(tum_time_diff[i], axis=1)
        tum_time_diff_mean.append(time_diff_mean)

    tum_time_diff_mean = np.array(tum_time_diff_mean)
    
    with open('./results/tum_time_diff_mean.pkl', 'wb') as f:
        pickle.dump(tum_time_diff_mean[:,1:4], f) 



    import numpy as np
    import pickle
    
    with open('./results/correct/C_all_correct.pkl', 'rb') as f:
        C_all_correct = pickle.load(f)
        
    with open('./results/error/C_all_error.pkl', 'rb') as f:
        C_all_error = pickle.load(f)          


    C_all = C_all_correct + C_all_error  # All

    with open('./results/C_all.pkl', 'wb') as f:
        pickle.dump(C_all, f) 




    C_all_mean = []
    for i in range(len(C_all)):
        C_all_one_mean = np.mean(C_all[i], axis=4)
        C_all_mean.append(C_all_one_mean)
        
    C_all_mean =np.array(C_all_mean)

    with open('./results/C_all_mean.pkl', 'wb') as f:
        pickle.dump(C_all_mean, f)




    #%%% 
    import numpy as np
    import pickle
    
    with open('./results/correct/C_all_correct_FF.pkl', 'rb') as f:
        C_all_correct = pickle.load(f)
        
    with open('./results/error/C_all_error_FF.pkl', 'rb') as f:
        C_all_error = pickle.load(f)          


    C_all_FF = C_all_correct + C_all_error  # All Follower-Follower
    with open('./results/C_all_FF.pkl', 'wb') as f:
        pickle.dump(C_all_FF, f) 



    C_all_mean = []
    for i in range(len(C_all_FF)):
        C_all_one_mean = np.mean(C_all_FF[i], axis=4)
        C_all_mean.append(C_all_one_mean)
        
    C_all_mean =np.array(C_all_mean)
    
    
    with open('./results/C_all_mean.pkl', 'rb') as file_obj:
        C_all_mean_LF = pickle.load(file_obj)
    

    C_index = np.concatenate((C_all_mean_LF, C_all_mean), axis=2)
    
    
    C_index = np.swapaxes(C_index, 0, 1) # 交换轴1和轴2  轴1为不同阶段
    C_index = np.swapaxes(C_index, 2, 4) 
    C_index = np.swapaxes(C_index, 2, 3) 
    

    with open('./results/C_index.pkl', 'wb') as f:
        pickle.dump(C_index, f)























