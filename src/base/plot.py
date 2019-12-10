"""
Author: Ian Osband
https://github.com/iosband/ts_tutorial/

Common scripts for plotting/analysing the results of experiments.

This code is designed to work with the .csv files that are output by
`batch_runner.py`.

Some of this code is generic, but a lot of it is designed specifically to
generate the plots that are used in the TS tutorial paper. For usage in
generating these plots see `batch_analysis.py`.
"""

from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import plotnine as gg

sys.path.append(os.getcwd())
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(10, 8))

_DEFAULT_DATA_PATH = '../log/data'
_DATA_CACHE = {}

#############################################################################
# Loading data

def set_data_path(file_path):
  """Overwrite globale default data path."""
  _DEFAULT_DATA_PATH = file_path


def _name_cleaner(agent_name):
  """Renames agent_name to prettier string for plots."""
  rename_dict = {'reinf': 'Policy Gradient REINFORCE',
                 'ac': 'Policy Gradient Actor-Critic',
                 'bbq_k4': 'BBQ k=0.4',
                 'bbq_k5': 'BBQ k=0.5',
                 'bbq_k8': 'BBQ k=0.8',
                 'bbq_k25': 'BBQ k=0.25',
                 'bbq_k3': 'BBQ k=0.3',
                 'ts': 'TS',
                 'greedy': 'Greedy'}
  if agent_name in rename_dict:
    return rename_dict[agent_name]
  else:
    return agent_name


def load_data(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Function to load in the data relevant to a specific experiment.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    df: dataframe of experiment data (uses cache for faster reloading).
  """
  if experiment_name in _DATA_CACHE:
    return _DATA_CACHE[experiment_name]
  else:
    all_files = os.listdir(data_path)
    good_files = []
    for file_name in all_files:
      if '.csv' not in file_name:
        continue
      else:
        file_experiment = file_name.split('exp=')[1].split('|')[0]
        if file_experiment == experiment_name:
          good_files.append(file_name)

    data = []
    for file_name in good_files:
      file_path = os.path.join(data_path, file_name)
      if 'id=' in file_name:
        if os.path.getsize(file_path) < 1024:
          continue
        else:
          data.append(pd.read_csv(file_path))
      elif 'params' in file_name:
        params_df = pd.read_csv(file_path)
        params_df['agent'] = params_df['agent'].apply(_name_cleaner)
      else:
        raise ValueError('Something is wrong with file names.')

    df = pd.concat(data)
    df = pd.merge(df, params_df, on='unique_id')
    _DATA_CACHE[experiment_name] = df
    return _DATA_CACHE[experiment_name]


#############################################################################
# Basic instant regret plots


def simple_algorithm_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'instant_regret': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'instant_regret', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('per-period regret')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
  
  plot_dict = {experiment_name + '_regret': p}
  return plot_dict

def cumulative_reward_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'cum_reward': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'cum_reward', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('cumulative reward')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
  
  plot_dict = {experiment_name + '_cum_reward': p}
  return plot_dict

def num_query_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'num_query': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'num_query', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('num queries')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
  
  plot_dict = {experiment_name + '_query': p}
  return plot_dict

def avg_reward_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'avg_reward': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'avg_reward', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('average reward')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
  
  plot_dict = {experiment_name + '_avg_reward': p}
  return plot_dict

