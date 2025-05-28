from typing import Dict

import relax
from pathlib import Path
import re
import csv
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context(font_scale=1.2)


def plot_mean(patterns_dict: Dict, env_name, fig_name = None,
              max_steps=None):
    plt.figure(figsize=(4, 3))
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    dfs = []
    for alg, pattern in patterns_dict.items():
        matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
        for dir in matching_dir:
            csv_path = dir / 'log.csv'
            df = pd.read_csv(str(csv_path))
            df.loc[:, ('seed')] = str(dir).split('_s')[1].split('_')[0]
            df.loc[:, ('alg')] = alg
            dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    if max_steps is not None:
        total_df = total_df[total_df['step'] < max_steps]
    sns.lineplot(data=total_df, x='step', y='avg_ret', hue='alg')
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
    


def load_best_results(pattern, env_name, show_df=False,
              max_steps=None):
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    # pattern = r".*diffv2.*noise_scale_0\.0\d$"
    # pattern = r".*diffv2.*noise_scale_0\.09"
    # pattern = r".*qsm.*01-07.*qsm_lr_schedule$"
    
    matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
    dfs = []
    for dir in matching_dir:
        csv_path = dir / 'log.csv'
        df = pd.read_csv(str(csv_path))
        if max_steps is not None:
            df = df[df['step'] < max_steps]
        sliced_df = df.loc[df['avg_ret'].idxmax()]
        sliced_df.loc['seed'] = str(dir).split('_s')[1].split('_')[0]
        # if 'lr_end' in dir:
        #     sliced_df.loc['lr_end'] = dir.split('lr_end_')[1]
        dfs.append(sliced_df)
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    if show_df:
        print(total_df.to_markdown())
    print(f"${total_df['avg_ret'].mean():.3f} \pm {total_df['avg_ret'].std():.3f}$")
    return total_df

if __name__ == "__main__":
    # pattern = r".*diffv2.*01-07.*diffv2_ema$"
    # load_best_results(pattern)
    # patterns_dict = {
    #                 #  'ema': r".*diffv2.*01-07.*diffv2_ema$",
    #                  'sampling_ema': r".*diffv2.*01-07.*diffv2_sampling_with_ema$",
    #                 #  'lr_schedule': r".*diffv2.*01-07.*diffv2_lr_schedule$", 
    #                 #  'qsm_lr': r".*qsm.*01-07.*qsm_lr_schedule$",
    #                  'qsm': r".*qsm.*01-07.*atp1$"}
    patterns_dict = {
        'sampling_ema': r".*diffv2.*01.*diffv2_sampling_with_ema$",
        # 'qsm': r".*qsm.*01.*atp1$",
        # 'sac': r".*sac.*01.*atp1$"
    }
    plot_mean(patterns_dict, 'Ant-v4')
