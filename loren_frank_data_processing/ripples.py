import pandas as pd

from .core import get_data_structure
from .tetrodes import get_LFP_dataframe


def _get_computed_ripple_times(tetrode_tuple, animals):
    '''Returns a list of tuples for a given tetrode in the format
    (start_index, end_index). The indexes are relative
    to the trial time for that session. Data is extracted from the ripples
    data structure and calculated according to the Frank Lab criterion.
    '''
    animal, day, epoch, tetrode_number = tetrode_tuple
    ripples_data = get_data_structure(
        animals[animal], day, 'ripples', 'ripples')
    return zip(
        ripples_data[epoch - 1][0][tetrode_number - 1]['starttime'][
            0, 0].flatten(),
        ripples_data[epoch - 1][0][tetrode_number - 1]['endtime'][
            0, 0].flatten())


def _convert_ripple_times_to_dataframe(ripple_times, dataframe):
    '''Given a list of ripple time tuples (ripple #, start time, end time)
    and a dataframe with a time index (such as the lfp dataframe), returns
    a pandas dataframe with a column with the timestamps of each ripple
    labeled according to the ripple number. Non-ripple times are marked as
    NaN.
    '''
    try:
        index_dataframe = dataframe.drop(dataframe.columns, axis=1)
    except AttributeError:
        index_dataframe = dataframe[0].drop(dataframe[0].columns, axis=1)
    ripple_dataframe = (pd.concat(
        [index_dataframe.loc[start_time:end_time].assign(
            ripple_number=number)
         for number, start_time, end_time in ripple_times]))
    try:
        ripple_dataframe = pd.concat(
            [dataframe, ripple_dataframe], axis=1,
            join_axes=[index_dataframe.index])
    except TypeError:
        ripple_dataframe = pd.concat(
            [pd.concat(dataframe, axis=1), ripple_dataframe],
            axis=1, join_axes=[index_dataframe.index])
    return ripple_dataframe


def get_computed_ripples_dataframe(tetrode_key, animals):
    '''Pre-computed ripples from the Frank lab labeled according to the
    ripple number with non-ripple times are marked as NaN.

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
    ripple_indicator : pandas.DataFrame

    '''
    ripple_times = _get_computed_ripple_times(tetrode_key, animals)
    [(ripple_ind + 1, start_time, end_time)
     for ripple_ind, (start_time, end_time) in enumerate(ripple_times)]
    lfp_dataframe = get_LFP_dataframe(tetrode_key, animals)
    return (_convert_ripple_times_to_dataframe(ripple_times, lfp_dataframe)
            .assign(
                ripple_indicator=lambda x: x.ripple_number.fillna(0) > 0))


def get_computed_consensus_ripple_times(epoch_key, animals):
    '''Returns a list of tuples for a given epoch in the format
    (start_time, end_time).

    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying the recording epoch. Elements are
        (animal_short_name, day, epoch).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    ripple_times : list of tuples
        Each list element corresponds to a ripple with (start_time, end_time).
    '''
    animal, day, epoch = epoch_key
    ripples_data = get_data_structure(
        animals[animal], day, 'candripples', 'candripples')
    return list(map(tuple, ripples_data[epoch - 1]['riptimes'][0][0]))
