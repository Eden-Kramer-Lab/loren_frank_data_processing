from os.path import join

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .core import logger
from .tetrodes import get_trial_time


def get_multiunit_dataframe(tetrode_key, animals, datastruct='by_tetrode'):
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
    animal, day, epoch, ntrode_num = tetrode_key

    if datastruct == 'by_tetrode':
        try:
            multiunit_file = loadmat(get_multiunit_filename(tetrode_key, animals, datastruct=datastruct))
        except (FileNotFoundError, TypeError):
            logger.warning('Failed to load file: {0}'.format(
                get_multiunit_filename(tetrode_key, animals)))
        multiunit_names = [
            name[0][0].lower().replace(' ', '_')
            for name in multiunit_file['filedata'][0, 0]['paramnames']]
        multiunit_data = multiunit_file['filedata'][0, 0]['params']
        time = pd.TimedeltaIndex(
            multiunit_data[:, multiunit_names.index('time')].astype(int) *
            TO_NANOSECONDS, unit='ns', name='time')
        return pd.DataFrame(
            multiunit_data, columns=multiunit_names,
            index=time).drop('time', axis=1)
    elif datastruct == 'by_day':
        try:
            # print(tetrode_key)
            multiunit_file = loadmat(get_multiunit_filename(tetrode_key, animals, datastruct=datastruct), squeeze_me=True,struct_as_record=False)
            multiunit_data = multiunit_file['marks'][epoch-1][ntrode_num-1].marks
        except (FileNotFoundError, TypeError, AttributeError):
            logger.warning('Failed to load marks for {filename} ntrode:{ntrode_num}'.format(filename=
                get_multiunit_filename(tetrode_key, animals, datastruct=datastruct), ntrode_num=ntrode_num))
            return

        time = pd.TimedeltaIndex(multiunit_file['marks'][epoch-1][ntrode_num-1].times, unit='s', name='time')
        num_channels = multiunit_data[0].shape[0]
        multiunit_names = ['channel_{chan:02d}_max'.format(chan=chan) for chan in range(0,num_channels)]
        return pd.DataFrame(
            multiunit_data, columns=multiunit_names, index=time)


def get_multiunit_filename(tetrode_key, animals, datastruct='by_tetrode'):
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
    if datastruct == 'by_tetrode':
        animal, day, _, tetrode_number = tetrode_key
        filename = ('{animal.short_name}marks{day:02d}-'
                    '{tetrode_number:02d}.mat').format(
            animal=animals[animal],
            day=day,
            tetrode_number=tetrode_number
        )
        return join(animals[animal].directory, 'EEG', filename)

    elif datastruct == 'by_day':
        animal, day, epoch, tetrode_number = tetrode_key
        print(tetrode_key)
        filename = '{animal.short_name}marks{day:02d}.mat'.format(animal=animals[animal], day=day)
        return join(animals[animal].directory, filename)

def get_multiunit_indicator_dataframe(tetrode_key, animals,
                                      time_function=get_trial_time, datastruct='by_tetrode'):
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
    multiunit_dataframe = (get_multiunit_dataframe(tetrode_key, animals, datastruct=datastruct)
                           .loc[time.min():time.max()])
    time_index = np.digitize(multiunit_dataframe.index.total_seconds(),
                             time.total_seconds())
    return (multiunit_dataframe.groupby(time[time_index]).mean()
            .reindex(index=time))
