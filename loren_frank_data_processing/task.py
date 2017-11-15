from glob import glob
from os.path import join

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_task(file_name, animal):
    '''Loads task information for a specific day and converts it to a pandas
    DataFrame.

    Parameters
    ----------
    file_name : str
        Task file name for an animal and recording session day.
    animal : namedtuple
        Information about data directory for the animal.

    Returns
    -------
    task_information : pandas.DataFrame

    '''
    data = loadmat(file_name, variable_names=('task'))['task']
    day = data.shape[-1]
    epochs = data[0, -1][0]
    n_epochs = len(epochs)
    index = pd.MultiIndex.from_product(
        ([animal.short_name], [day], np.arange(n_epochs) + 1),
        names=['animal', 'day', 'epoch'])

    return pd.DataFrame(
        [{name: epoch[name].item().squeeze()
          for name in epoch.dtype.names
          if name in ['environment', 'type']}
         for epoch in epochs]).set_index(index).assign(
            environment=lambda df: df.environment.astype(str),
            type=lambda df: df.type.astype(str))


def compute_exposure(epoch_info):
    exposure = (epoch_info
                .groupby('animal').environment
                .apply(lambda s: pd.get_dummies(s).cumsum())
                .stack()
                .reset_index()
                .rename(columns={'level_3': 'environment', 0: 'exposure'})
                .set_index(['animal', 'day', 'epoch', 'environment']))

    epoch_keys = epoch_info.set_index('environment', append=True).index
    new_df = (epoch_info.drop('environment', axis=1)
              .join(exposure.loc[epoch_keys].reset_index('environment')))
    new_df['exposure'] = new_df.exposure.where(
        ~epoch_info.type.isin(['sleep', 'rest', 'nan']))
    return new_df


def get_task(animal):
    '''Loads all experimental information for all days for a given animal.

    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    task_information : pandas.DataFrame

    '''
    task_files = glob(join(animal.directory, '*task*.mat'))
    return pd.concat(load_task(task_file, animal)
                     for task_file in task_files)


def make_epochs_dataframe(animals):
    '''Experimental conditions for all recording epochs.

    Index is a unique identifying key for that recording epoch.

    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    epoch_information : pandas.DataFrame

    '''
    return compute_exposure(
        pd.concat([get_task(animal) for animal in animals.values()])
        .sort_index())
