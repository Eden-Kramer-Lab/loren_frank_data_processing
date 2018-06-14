'''Loads general information about each spike-sorted neuron, spike times, or
spike indicators.
'''

from os.path import join

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .core import _convert_to_dict, get_data_filename, logger
from .tetrodes import get_trial_time


def make_neuron_dataframe(animals):
    '''Information about all recorded neurons such as brain area.

    The index of the dataframe corresponds to the unique key for that neuron
    and can be used to load spiking information.

    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    neuron_information : pandas.DataFrame

    '''
    neuron_file_names = [(get_neuron_info_path(animals[animal]), animal)
                         for animal in animals]
    neuron_data = [(loadmat(file_name[0]), file_name[1])
                   for file_name in neuron_file_names]
    return pd.concat([
        convert_neuron_epoch_to_dataframe(
            epoch, animal, day_ind + 1, epoch_ind + 1)
        for cellfile, animal in neuron_data
        for day_ind, day in enumerate(cellfile['cellinfo'].T)
        for epoch_ind, epoch in enumerate(day[0].T)
    ]).sort_index()


def get_spikes_dataframe(neuron_key, animals):
    '''Spike times for a particular neuron.

    Parameters
    ----------
    neuron_key : tuple
        Unique key identifying that neuron. Elements of the tuple are
        (animal_short_name, day, epoch, tetrode_number, neuron_number).
        Key can be retrieved from `make_neuron_dataframe` function.
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    spikes_dataframe : pandas.DataFrame
    '''
    animal, day, epoch, tetrode_number, neuron_number = neuron_key
    try:
        neuron_file = loadmat(
            get_data_filename(animals[animal], day, 'spikes'))
    except (FileNotFoundError, TypeError):
        logger.warning('Failed to load file: {0}'.format(
            get_data_filename(animals[animal], day, 'spikes')))
    try:
        spike_time = neuron_file['spikes'][0, -1][0, epoch - 1][
            0, tetrode_number - 1][0, neuron_number - 1][0]['data'][0][
            :, 0]
        spike_time = pd.TimedeltaIndex(spike_time, unit='s', name='time')
    except IndexError:
        spike_time = []
    return pd.Series(
        np.ones_like(spike_time, dtype=int), index=spike_time,
        name='{0}_{1:02d}_{2:02}_{3:03}_{4:03}'.format(*neuron_key))


def get_spike_indicator_dataframe(neuron_key, animals,
                                  time_function=get_trial_time):
    '''A time series where 1 indicates a spike at that time and 0 indicates no
    spike at that time.

    Parameters
    ----------
    neuron_key : tuple
        Unique key identifying that neuron. Elements of the tuple are
        (animal_short_name, day, epoch, tetrode_number, neuron_number).
        Key can be retrieved from `make_neuron_dataframe` function.
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    time_function : function, optional
        Function that take an epoch key (animal_short_name, day, epoch) that
        defines the time the multiunits are relative to. Defaults to using
        the time the LFPs are sampled at.

    Returns
    ---

    '''
    time = time_function(neuron_key[:3], animals)
    spikes_df = get_spikes_dataframe(neuron_key, animals)
    time_index = np.digitize(spikes_df.index.total_seconds(),
                             time.total_seconds())
    return (spikes_df.groupby(time[time_index]).sum()
            .reindex(index=time, fill_value=0))


def get_all_spike_indicators(neuron_keys, animals,
                             time_function=get_trial_time):
    time = time_function(neuron_keys[0][:3], animals)
    spikes_dfs = [get_spikes_dataframe(neuron_key, animals)
                  for neuron_key in neuron_keys]
    return pd.concat(
        (spikes_df
         .groupby(time[np.digitize(spikes_df.index.total_seconds(),
                                   time.total_seconds())]).sum()
         .reindex(index=time, fill_value=0)
         for spikes_df in spikes_dfs), axis=1)


def convert_neuron_epoch_to_dataframe(tetrodes_in_epoch, animal, day,
                                      epoch):
    '''
    Given an neuron data structure, return a cleaned up DataFrame
    '''
    DROP_COLUMNS = ['ripmodtag', 'thetamodtag', 'runripmodtag',
                    'postsleepripmodtag', 'presleepripmodtag',
                    'runthetamodtag', 'ripmodtag2', 'runripmodtag2',
                    'postsleepripmodtag2', 'presleepripmodtag2',
                    'ripmodtype', 'runripmodtype', 'postsleepripmodtype',
                    'presleepripmodtype', 'FStag', 'ripmodtag3',
                    'runripmodtag3', 'ripmodtype3', 'runripmodtype3',
                    'tag', 'typetag', 'runripmodtype2',
                    'tag2', 'ripmodtype2', 'descrip']

    NEURON_INDEX = ['animal', 'day', 'epoch',
                    'tetrode_number', 'neuron_number']

    neuron_dict_list = [_add_to_dict(
        _convert_to_dict(neuron), tetrode_ind, neuron_ind)
        for tetrode_ind, tetrode in enumerate(
        tetrodes_in_epoch[0][0])
        for neuron_ind, neuron in enumerate(tetrode[0])
        if neuron.size > 0
    ]
    try:
        return (pd.DataFrame(neuron_dict_list)
                  .drop(DROP_COLUMNS, axis=1, errors='ignore')
                  .assign(animal=animal)
                  .assign(day=day)
                  .assign(epoch=epoch)
                  .assign(neuron_id=_get_neuron_id)
                # set index to identify rows
                  .set_index(NEURON_INDEX)
                  .sort_index()
                )
    except AttributeError:
        logger.warn('{0}, {1}, {2} not processed'.format(animal, day, epoch))


def get_neuron_info_path(animal):
    '''Returns the path to the neuron info matlab file

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.

    Returns
    -------
    path : str

    '''
    filename = '{animal.short_name}cellinfo.mat'.format(animal=animal)
    return join(animal.directory, filename)


def _get_neuron_id(dataframe):
    '''Unique identifier string for a neuron'''
    return (dataframe.animal + '_' +
            dataframe.day.map('{:02d}'.format) + '_' +
            dataframe.epoch.map('{:02}'.format) + '_' +
            dataframe.tetrode_number.map('{:03}'.format) + '_' +
            dataframe.neuron_number.map('{:03}'.format))


def _add_to_dict(dictionary, tetrode_ind, neuron_ind):
    dictionary['tetrode_number'] = tetrode_ind + 1
    dictionary['neuron_number'] = neuron_ind + 1
    return dictionary
