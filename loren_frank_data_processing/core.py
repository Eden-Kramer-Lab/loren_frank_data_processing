'''Functions for accessing data in the Frank lab format and saving

'''
from collections import namedtuple
from logging import getLogger
from os.path import join

import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = getLogger(__name__)

Animal = namedtuple('Animal', {'directory', 'short_name'})


def get_data_filename(animal, day, file_type):
    '''Returns the Matlab file name assuming it is in the Raw Data
    directory.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of recording
    file_type : str
        Data structure name (e.g. linpos, dio)

    Returns
    -------
    filename : str
        Path to data file

    '''
    filename = '{animal.short_name}{file_type}{day:02d}.mat'.format(
        animal=animal,
        file_type=file_type,
        day=day)
    return join(animal.directory, filename)


def get_epochs(animal, day):
    '''For a given recording day and animal, get the three-element epoch
    key that uniquely identifys the recording epochs in that day.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of the recording.

    Returns
    -------
    epochs : list of tuples, shape (n_epochs,)
         A list of three-element tuples (animal, day, epoch key) that
         uniquely identifys the recording epochs in that day.

    Examples
    --------
    >>> from collections import namedtuple
    >>> Animal = namedtuple('Animal', {'directory', 'short_name'})
    >>> animal = Animal(directory='test_dir', short_name='Test')
    >>> day = 2
    >>> get_epochs(animal, day)

    '''
    try:
        task_file = loadmat(
            get_data_filename(animal, day, 'task'))
        return [(animal, day, ind + 1)
                for ind, epoch in enumerate(task_file['task'][0, -1][0])]
    except (IOError, TypeError) as err:
        logger.warn('Failed to load file {0}'.format(
            get_data_filename(animal, day, 'task')))
        exit()


def get_data_structure(animal, day, file_type, variable):
    '''Returns data structures corresponding to the animal, day, file_type
    for all epochs

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of recording
    file_type : str
        Data structure name (e.g. linpos, dio)
    variable : str
        Variable in data structure

    Returns
    -------
    variable : list, shape (n_epochs,)
        Elements of list are data structures corresponding to variable

    '''
    try:
        file = loadmat(get_data_filename(animal, day, file_type))
        n_epochs = file[variable][0, -1].size
        return [file[variable][0, -1][0, ind]
                for ind in np.arange(n_epochs)]
    except (IOError, TypeError):
        logger.warn('Failed to load file: {0}'.format(
            get_data_filename(animal, day, file_type)))
        return None


def reconstruct_time(start_time, n_samples, sampling_frequency):
    '''Reconstructs the recording time

    Parameters
    ----------
    start_time : float
        Start time of recording.
    n_samples : int
        Number of samples in recording.
    sampling_frequency : float
        Number of samples per time

    Returns
    -------
    time : pandas Index

    '''
    return pd.TimedeltaIndex(
        start=pd.Timedelta(start_time, unit='s'),
        end=pd.Timedelta(start_time + (n_samples - 1) / sampling_frequency,
                         unit='s'),
        periods=n_samples, unit='s', name='time')


def _convert_to_dict(struct_array):
    try:
        return {name: struct_array[name].item().item()
                for name in struct_array.dtype.names
                if struct_array[name].item().size == 1}
    except TypeError:
        return {}
