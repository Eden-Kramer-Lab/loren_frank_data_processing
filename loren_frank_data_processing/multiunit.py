from os.path import join

import dask
import numpy as np
import pandas as pd
import xarray as xr
from loren_frank_data_processing.core import get_data_filename, logger
from loren_frank_data_processing.tetrodes import get_trial_time
from scipy.io import loadmat


def get_multiunit_dataframe(tetrode_key, animals):
    '''Retrieve the multiunits for each tetrode given a tetrode key

    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    multiunit_dataframe : pandas dataframe
        The dataframe index is the time at which the multiunit occurred
        (in seconds). THe other values are values that can be used as the
        multiunits.

    '''
    TO_NANOSECONDS = 1E5
    try:
        multiunit_file = loadmat(get_multiunit_filename(tetrode_key, animals))
        multiunit_names = [
            name[0][0].lower().replace(' ', '_')
            for name in multiunit_file['filedata'][0, 0]['paramnames']]
        multiunit_data = multiunit_file['filedata'][0, 0]['params']
        time = pd.TimedeltaIndex(
            multiunit_data[:, multiunit_names.index('time')].astype(int) *
            TO_NANOSECONDS, unit='ns', name='time')
        is_duplicated = time.duplicated()

        return (pd.DataFrame(multiunit_data, columns=multiunit_names,
                             index=time)
                .drop('time', axis=1)
                .loc[~is_duplicated]
                .sort_index())
    except (FileNotFoundError, TypeError):
        pass


def get_multiunit_dataframe2(tetrode_key, animals):
    '''Retrieve the multiunits for each tetrode given a tetrode key
    This function retrieves data according to Demetris's format.

    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    multiunit_dataframe : pandas dataframe
        The dataframe index is the time at which the multiunit occurred
        (in seconds). THe other values are values that can be used as the
        multiunits.

    '''
    animal, day, epoch, tetrode_number = tetrode_key
    try:
        multiunit_file = loadmat(
            get_data_filename(animals[animal], day, 'marks'))
        multiunit_data = multiunit_file['marks'][0, -1][0, epoch - 1][
            0, tetrode_number - 1]

        time = pd.TimedeltaIndex(multiunit_data['times'][0, 0].squeeze(),
                                 unit='s', name='time')
        multiunit = multiunit_data['marks'][0, 0].astype(np.float)
        column_names = ['channel_{number}_max'.format(number=number + 1)
                        for number in np.arange(multiunit.shape[1])]
        is_duplicated = time.duplicated()
        return (pd.DataFrame(multiunit, columns=column_names, index=time)
                .loc[~is_duplicated]
                .sort_index())
    except (FileNotFoundError, TypeError):
        logger.warning('Failed to load file: {0}'.format(
            get_data_filename(animals[animal], day, 'marks')))


def get_multiunit_filename(tetrode_key, animals):
    '''Path for the multiunits for a particular tetrode.

    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    multiunit_filename : str

    '''
    animal, day, _, tetrode_number = tetrode_key
    filename = ('{animal.short_name}marks{day:02d}-'
                '{tetrode_number:02d}.mat').format(
        animal=animals[animal],
        day=day,
        tetrode_number=tetrode_number
    )
    return join(animals[animal].directory, 'EEG', filename)


def get_multiunit_indicator_dataframe(tetrode_key, animals,
                                      time_function=get_trial_time):
    '''A time series where a value indicates multiunit activity at that time and
    NaN indicates no multiunit activity at that time.

    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    time_function : function, optional
        Function that take an epoch key (animal_short_name, day, epoch) that
        defines the time the multiunits are relative to. Defaults to using
        the time the LFPs are sampled at.

    Returns
    -------
    multiunit_indicator : pandas.DataFrame, shape (n_time, n_features)

    '''
    time = time_function(tetrode_key[:3], animals)
    try:
        multiunit_dataframe = (get_multiunit_dataframe(tetrode_key, animals)
                               .loc[time.min():time.max()])
    except AttributeError:
        multiunit_dataframe = get_multiunit_dataframe2(tetrode_key, animals)
    time_index = np.digitize(multiunit_dataframe.index.total_seconds(),
                             time.total_seconds()[1:-1])
    return (multiunit_dataframe.groupby(time[time_index]).mean()
            .reindex(index=time))


@dask.delayed
def _get_indicator(tetrode_key, animals, time):
    try:
        df = (get_multiunit_dataframe(tetrode_key, animals)
              .loc[time.min():time.max()])
    except AttributeError:
        df = (get_multiunit_dataframe2(tetrode_key, animals)
              .loc[time.min():time.max()])

    time_index = np.digitize(df.index.total_seconds(),
                             time.total_seconds()[1:-1])

    return (df.groupby(time[time_index]).mean()
            .reindex(index=time)
            .to_xarray()
            .to_array('features'))


def get_all_multiunit_indicators(tetrode_keys, animals,
                                 time_function=get_trial_time):
    time = time_function(tetrode_keys[0][:3], animals)
    multiunit_dfs = []

    for tetrode_key in tetrode_keys:
        multiunit_dfs.append(_get_indicator(tetrode_key, animals, time))

    return (xr.concat(
        dask.compute(*multiunit_dfs, scheduler='threads'), dim='tetrodes')
        .transpose('time', 'features', 'tetrodes'))
