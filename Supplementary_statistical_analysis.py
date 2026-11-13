"""Group-level ANOVA, Tukey comparisons and supplementary summaries."""
# Original author: Chen Min

import numpy as np
import pickle
from pathlib import Path

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def data_group():
    """Group LF IBC by dyad-level drumbeat time differences."""

    with open('./results/C_all_mean.pkl', 'rb') as file_obj:
        C_all_mean = pickle.load(file_obj)

    C_all_mean = np.swapaxes(C_all_mean, 0, 1) # Move the stage axis to the first position.

    with open('./results/tum_time_diff_mean.pkl', 'rb') as file_obj:
        tum_time_diff = pickle.load(file_obj)

    # Group observations by the behavioral gold standard.
    good_p = np.array(np.where(tum_time_diff<0.4)).T # Good: MTD < 0.4 s.
    middle_p = np.array(np.where((tum_time_diff>0.4) & (tum_time_diff<0.55))).T # 0.4s<T<0.55
    bad_p = np.array(np.where(tum_time_diff>0.55)).T  # T>0.55s

    C_good = C_all_mean[:,good_p[:,0],good_p[:,1],:,:,:,:] # Good
    C_middle = C_all_mean[:,middle_p[:,0],middle_p[:,1],:,:,:,:] # Middle
    C_bad = C_all_mean[:,bad_p[:,0],bad_p[:,1],:,:,:,:] # Bad

    return C_good, C_middle, C_bad


def anova_analysis(data):
    """Run one-way ANOVA followed by Tukey HSD when significant."""

    df_data = pd.DataFrame(data).T
    df_data.columns = ['G', 'M', 'B']
    df_data_melt = df_data.melt()
    df_data_melt = df_data_melt.dropna() # Drop missing values.
    df_data_melt.columns = ['Class','Index']

    model = ols('Index~C(Class)', data=df_data_melt).fit()
    anova_table = anova_lm(model)

    f_value = anova_table['F'].iloc[0]
    p_value = anova_table['PR(>F)'].iloc[0]

    # Tukey pairwise comparisons.
    if p_value<0.05:
        tukey_result = pairwise_tukeyhsd(df_data_melt['Index'], df_data_melt['Class'], alpha = 0.05)
        tukey_result_ = np.zeros((3,3))
        tukey_result_[:,0] = tukey_result.reject.astype(np.int32)
        tukey_result_[:,1::] = tukey_result.confint
    else:
        tukey_result_ = np.zeros((3,3))

    return p_value, f_value, tukey_result_


def statis_analysis_group():
    """Compare coordination groups for each stage, metric, band and channel pair."""

    C_good, C_middle, C_bad = data_group()

    p_value_all = np.zeros((3, 6, 5, 30, 30))
    f_value_all = np.zeros((3, 6, 5, 30, 30))
    tukey_result_all = np.zeros((3, 6, 5, 30, 30, 3, 3))

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


def sig3(x):
    return [float(f"{v:.3g}") for v in x]


def statistic_results(results):

    p_value = results['p_value']
    f_value = results['F_value']
    tukey_results = results['tukey']

    stage = ['Pre', 'Exp', 'Post']
    index1 = ['CCorr', 'Coh', 'PLI', 'WPLI', 'PowerCorr', 'PLV']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] # Frequency bands

    r_all = {}

    for s in range(3):
        r_all[stage[s]] = {}
        for i in range(6):
            for j in range(5):

                r_all[stage[s]][f'{index1[i]}_{bands[j]}'] = {}
                p = p_value[s,i,j,:]
                index_ = np.where(p<0.05)
                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['N'] = f'{len(index_[0])}/900'

                f = f_value[s,i,j,:]
                f = f[index_]
                t = tukey_results[s,i,j,:]
                t = t[index_]

                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['F_median'] = f'F = {np.median(f):.3g}'

                index_GB = np.where(t[:,0,0] == 1)
                index_MB = np.where(t[:,1,0] == 1)
                index_GM = np.where(t[:,2,0] == 1)

                ci = t[index_GB][:,0,1:]
                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['GB_CI_median'] = sig3(np.median(ci, axis=0))

                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['GB_N'] = f'N = {len(index_GB[0])}'

                ci = t[index_MB][:,1,1:]
                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['MB_CI_median'] = sig3(np.median(ci, axis=0))

                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['MB_N'] = f'N = {len(index_MB[0])}'

                ci = t[index_GM][:,2,1:]
                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['GM_CI_median'] = sig3(np.median(ci, axis=0))

                r_all[stage[s]][f'{index1[i]}_{bands[j]}']['GM_N'] = f'N = {len(index_GM[0])}'

    return r_all

#%% main

def main(output_dir="results/statis_abalysis/group"):
    """Save group statistics and supplementary summaries in one output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p_value_all, f_value_all, tukey_result_all = statis_analysis_group()

    results = {'p_value':p_value_all, 'F_value': f_value_all, 'tukey':tukey_result_all}

    with (output_dir / 'Supplementary_statistical_resutls.py.pkl').open('wb') as f:
        pickle.dump(results, f)


    r_all = statistic_results(results)

    df = pd.DataFrame(r_all['Pre']).T
    df.to_excel(output_dir / "output_Pre.xlsx", index=True)

    df = pd.DataFrame(r_all['Exp']).T
    df.to_excel(output_dir / "output_Exe.xlsx", index=True)

    df = pd.DataFrame(r_all['Post']).T
    df.to_excel(output_dir / "output_Post.xlsx", index=True)
    return results


if __name__ == '__main__':
    main()
