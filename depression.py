"""Preprocess the clinical extension and apply the selected SVM."""
# Original author: comeo

#%%
import os
from NDFSysMNE import mneNDF
import mne
import numpy as np
from pathlib import Path
import hypyp
from spkit.eeg import ATAR
import pickle


def find_file_dir(root_dir):
    """List recording subdirectories in their existing order."""
    return [str(folder) for folder in Path(root_dir).iterdir() if folder.is_dir()]


def data_input(file_path):
    """Select EEG channels, filter 0.5-50 Hz, resample to 200 Hz, denoise and average-reference."""
    ndfMneObj = mneNDF(file_path)
    data = ndfMneObj.read2MneRaw()

    include_ch = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5',
                   'FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6',
                   'Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']

    # Select EEG channels.
    data_sel = data.pick(include_ch)
    # Apply a 0.5-50 Hz band-pass filter.
    data_sel = data_sel.filter(l_freq=0.5, h_freq=50, verbose=False)
    data_sel.resample(sfreq=200)  # Resample to 200 Hz.

    info = mne.create_info(ch_names=include_ch, sfreq=200, ch_types='eeg')
    # Wavelet denoising.
    X = data_sel.get_data().T * 10**6  # Transpose and convert volts to microvolts.
    XR = ATAR(X, wv='db3', winsize=200, beta=0.3, IPR=[25,75], OptMode='soft')
    data_denosing = mne.io.RawArray(XR.T*10**-6, info) # Reconstruct an MNE Raw object.

    # Set standard electrode positions.
    data_denosing.set_montage(montage="standard_1020")

    data_denosing.set_eeg_reference(ref_channels='average') # Apply the average reference.

    return data_denosing


def effective_tum(everygroup_dir):
    """Detect valid drumbeat times for the recorded participants."""

    time_group = [] # Drumbeat times for the recorded participants.
    for everyone_dir in everygroup_dir[0:2]:

        datatum, times =  mneNDF(everyone_dir).read2MneRaw().filter(l_freq=2,
                h_freq=50, verbose=False).pick('GSR').resample(sfreq=200)[:]
        # Remove local offsets before detecting drumbeat peaks.
        datatum = datatum[0]
        sample_num = 50
        for i in range(len(datatum)//sample_num):
            datatum[(i*sample_num):(i+1)*sample_num] = datatum[(i*sample_num):\
                (i+1)*sample_num] - np.mean(datatum[(i*sample_num):(i+1)*sample_num] )
        datatum = np.abs(datatum)  # Use absolute amplitudes.

        # Find samples above the drumbeat threshold.
        flag_tum = np.where(datatum>0.01)
        # Find the first sample of each separated peak.
        first_point = np.where(np.diff(flag_tum[0])>1000)[0]+1

        # Construct event times.
        event_times = np.append(flag_tum[0][0], flag_tum[0][first_point])

        # Identify missing drumbeats.
        missing_point = [] # Missing beat positions
        missing_values = [] # Values to insert
        diff_time = np.diff(event_times) # Time differences
        for index, diff_time_ in enumerate(diff_time):
            value_insert=[]
            for i in range(1, 11): # Allow up to ten consecutive missing beats.
                if diff_time_ > (i*2000+1000):
                    value_insert = [0] * i # Number of placeholders to insert

            if value_insert:
                missing_point.append(index+1)
                missing_values.append(np.array(value_insert))

        # Insert zeros for missing beats.
        event_sup = event_times[:]
        num = 0
        for i in range(len(missing_point)):
            event_sup = np.insert(event_sup, missing_point[i]+num, missing_values[i])
            num = num + len(missing_values[i])

        time_group.append(event_sup)

    time_group = np.array(time_group) # Convert to an array.
    time_group = time_group[:, 0:48] # Keep the first eight minutes.

    return time_group


def tum_standard(full_test):
    """Compute drumbeat timing differences as the behavioral gold standard."""
    correct_group = [] # Teams processed successfully
    error_group = [] # Teams with processing errors
    tum_time_diff = [] # Store drumbeat time differences.
    time_group_all = [] # Store drumbeat times.

    for single_test in full_test:
        everygroup_dir = find_file_dir(single_test) # Recording directories for this team

        try:
            time_group = effective_tum(everygroup_dir)
            correct_group.append(single_test)
            c_en = True
        except (ValueError, AttributeError, IndexError) as e:
            print(f"{e}"+': '+ single_test)
            error_group.append(single_test) # Teams with processing errors
            c_en = False

        if c_en:
            time_diff_all = [] # Store within-team timing differences.
            remove_time = np.where(time_group==0)
            time_group = np.delete(time_group, remove_time[1], axis=1) # Remove trials with missing drumbeats.
            time_group_all.append(time_group) # Store valid drumbeat times.

            for i in range(0,2):
                time_diff = (time_group[i] - time_group[0])/200   # Time difference in seconds
                time_diff_all.append(time_diff)
            tum_time_diff.append(np.mean(time_diff_all, axis=1))

    return correct_group, time_group_all, tum_time_diff, error_group


def data_split(everygroup_dir, time_group):
    """Epoch preparation (-2.5 to -0.5 s), execution (-0.5 to 1.5 s) and recovery (1.5 to 3.5 s)."""
    epochs_group_pre = [] # Preparation epochs
    epochs_group_exp = [] # Execution epochs
    epochs_group_post = [] # Recovery epochs

    for everyone_dir in everygroup_dir[0:2]:
        data_denosing = data_input(everyone_dir) # Load and preprocess this participant.
        # Construct event times.
        event_times = time_group[0]
        event =  np.vstack((event_times, np.repeat(0, len(event_times)),
                                 np.repeat(1, len(event_times)))).T

        # Epoch and baseline-correct the recording.
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


def bulid_dataset(correct_group, time_group_all, delta_low=1):
    """Compute leader-patient IBC for each stage."""

    epochs_all_pre = [] # Store EEG epochs.
    epochs_all_exp = [] # Store EEG epochs.
    epochs_all_post = [] # Store EEG epochs.

    C_all  = [] # Store IBC metrics.
    error_group_index = [] # Indices of teams with processing errors

    for index, single_test in enumerate(correct_group):
        everygroup_dir = find_file_dir(single_test) # Recording directories for this team
        try:
            epochs_group_pre, epochs_group_exp, epochs_group_post = \
                    data_split(everygroup_dir, time_group_all[index]) # Epochs for the three stages
            epochs_all_pre.append(epochs_group_pre)
            epochs_all_exp.append(epochs_group_exp)
            epochs_all_post.append(epochs_group_post)

        except (ValueError, AttributeError, IndexError, RuntimeError) as e:
            print(f"{e}"+': '+ single_test)
            error_group_index.append(index) # Teams with processing errors
            continue  # Skip this team.

        freq_bands = {'delta':[delta_low, 4], 'theta':[4,8], 'alpha':[8,13],
                      'beta':[13,30], 'gamma':[30,49]}
        modes = [ 'ccorr', 'coh', 'pli', 'wpli', 'pow_corr', 'plv']  # IBC metrics

        n_ch = 30 # Number of EEG channels

        C_group = [] # IBC metrics for each stage
        for epochs_group in [epochs_group_pre, epochs_group_exp, epochs_group_post]:

            data = np.array(epochs_group)

            complex_signal = hypyp.analyses.compute_freq_bands(data=data,
                                 sampling_rate=200, freq_bands=freq_bands)
            C_mode = []
            for mode in modes:
                result = hypyp.analyses.compute_sync(complex_signal= \
                            complex_signal, mode=mode, epochs_average=True)

                C_mode.append(result[:, 0:n_ch, n_ch:2*n_ch])  # Retain the cross-participant channel block.

            C_group.append(C_mode)

        C_all.append(C_group)

    return epochs_all_pre, epochs_all_exp, epochs_all_post, C_all, error_group_index

#%% main

if __name__ == '__main__':

    root_dir = os.environ.get("EEG_CLINICAL_ROOT", "data_depression")  # Root recording directory
    full_test = find_file_dir(root_dir) # Individual experiment directories
    correct_group, time_group_correct, tum_time_diff_correct, error_group = tum_standard(full_test)

    epochs_all_pre, epochs_all_exp, epochs_all_post, C_all_D, \
          error_group_index = bulid_dataset(correct_group, time_group_correct)

    C_all_D1 = np.array(C_all_D)

    with open('./results/C_depression.pkl', 'wb') as f:
        pickle.dump(C_all_D1, f)

#%% Index

np.random.seed(2)
Path("results/figure/SHAP").mkdir(parents=True, exist_ok=True)

import shap
import pandas as pd
from sklearn import svm
from matplotlib import pyplot as plt

from feature_names import feature_names

config = {
    "savefig.dpi": 300,  # Figure resolution
    'font.size': 18,               # Default font size
    'font.family': 'Times New Roman',  # Font family
    'axes.titlesize': 18,          # Title font size
    'axes.labelsize': 18,          # Axis label font size
    'xtick.labelsize': 16,         # X-axis tick font size
    'ytick.labelsize': 16,         # Y-axis tick font size
    'legend.fontsize': 16,         # Legend font size
}
plt.rcParams.update(config)  # Apply plotting style.

with open('./results/C_depression.pkl', 'rb') as file_obj:
    C_depression = pickle.load(file_obj)

C_depression = np.swapaxes(C_depression, 0, 1) # Move the stage axis to the first position.
num_ = np.size(C_depression,1)
C_depression = C_depression.reshape((3,num_,-1))

C_D = {}

# Load the dataset.
with open('./results/results/mean/All_Stages/dataset_mean_all_stages.pkl', 'rb') as file_obj:
    dataset_mean_all_stages = pickle.load(file_obj)

X_train = dataset_mean_all_stages['X_train']
y_train = dataset_mean_all_stages['y_train']
X_test = dataset_mean_all_stages['X_test']
y_test = dataset_mean_all_stages['y_test']

# Feature names
with open('./results/results/mean/feat_index_mean.pkl', 'rb') as f:
    feat_index_mean =  pickle.load(f)

with open('./results/results/mean/feat_names_mean.pkl', 'rb') as file_obj:
    feat_names_mean = pickle.load(file_obj)

feat_names_all = feature_names() # Generate all feature names.

stages = ['pre', 'exp', 'post']

for i, stage in enumerate(stages):
    feat_index = feat_index_mean[stage]
    feat_names_mean[stage] = [feat_names_all[i] for i in feat_index] # Names of selected features

    C_D[stage] = np.array([C_depression[i,:,j] for j in feat_index]).T

C_D = np.concatenate((C_D['pre'],  C_D['exp'], C_D['post']), axis=1)

feat_names = [name+'_Pre' for name in feat_names_mean['pre']] + \
              [name+'_Exe' for name in feat_names_mean['exp']] + \
              [name+'_Rec' for name in feat_names_mean['post']]

X_train = pd.DataFrame(data = X_train, columns=feat_names)
X_test = pd.DataFrame(data = X_test, columns=feat_names)

C_D = pd.DataFrame(data = C_D, columns=feat_names)

# Fit the selected model.
with open('./results/results/mean/All_Stages/results_all_stages_tuning.pkl', 'rb') as file_obj:
    results_all_stages_tuning = pickle.load(file_obj)

# Explain SVM predictions.
index = results_all_stages_tuning['SVM']['index_optimal']
clf_SVM = svm.SVC(C=index[0], gamma=index[1], probability=True, random_state=2)

clf_SVM = clf_SVM.fit(X_train, y_train)

y_pred = clf_SVM.predict(C_D)

# Compute SHAP values.
explainer = shap.KernelExplainer(clf_SVM.predict_proba, shap.kmeans(X_train, 2), link="logit")
shap_values = explainer.shap_values(C_D, nsamples=500)


def bar_color(i):
    colors = ['#f59311', '#63b2ee', '#1F77B4']
    return colors[i]

shap.summary_plot(shap_values, C_D, plot_type="bar", class_names=['Bad','Middle','Good'],
                  max_display=20, show=False, class_inds=[2,1,0], color=bar_color)
plt.xlabel("Mean SHAP Value")

save_dir = './results/figure/SHAP/Depression.png'
plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()

shap.summary_plot(shap_values[:,:,0], C_D, max_display=15, show=False)
plt.gcf().set_size_inches(5, 6)  # Get the current figure.
plt.gcf().set_dpi(300)  # Set figure resolution.
plt.tight_layout(pad=1)  # Set layout padding.
plt.show()

save_dir = './results/figure/SHAP/Depression_bad.png'
plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
