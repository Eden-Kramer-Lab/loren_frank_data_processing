from os.path import join

import numpy as np
import pandas as pd
from loren_frank_data_processing.core import logger, reconstruct_time
from scipy.io import loadmat


def squeeze_matcell(x):
    try:
        return x[0, 0][0, 0]
    except IndexError:
        try:
            return x[0, 0][0]
        except IndexError:
            return x[0, 0]


def _convert_to_dict(tetrode):
    try:
        return {name: squeeze_matcell(tetrode[name])
                for name in tetrode.dtype.names}
    except TypeError:
        return {}


def get_tetrode_info_path(animal):
    '''Returns the Matlab tetrode info file name assuming it is in the
    Raw Data directory.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.

    Returns
    -------
    filename : str
        The path to the information about the tetrodes for a given animal.

    '''
    return join(animal.directory, f'{animal.short_name}tetinfo.mat')


def get_LFP_dataframe(tetrode_key, animals):
    '''Gets the LFP data for a given epoch and tetrode.

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
    LFP : pandas dataframe
        Contains the electric potential and time
    '''
    try:
        lfp_file = loadmat(get_LFP_filename(tetrode_key, animals))
        lfp_data = lfp_file['eeg'][0, -1][0, -1][0, -1]
        lfp_time = reconstruct_time(
            lfp_data['starttime'][0, 0].item(),
            lfp_data['data'][0, 0].size,
            float(lfp_data['samprate'][0, 0].squeeze()))
        return pd.Series(
            data=lfp_data['data'][0, 0].squeeze().astype(float),
            index=lfp_time,
            name='{0}_{1:02d}_{2:02}_{3:03}'.format(*tetrode_key))
    except (FileNotFoundError, TypeError):
        logger.warning('Failed to load file: {0}'.format(
            get_LFP_filename(tetrode_key, animals)))


def make_tetrode_dataframe(animals, epoch_key=None):
    """Information about all tetrodes such as recording location.

    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    tetrode_infomation : pandas.DataFrame

    """
    tetrode_info = []

    if epoch_key is not None:
        animal, day, epoch = epoch_key
        file_name = get_tetrode_info_path(animals[animal])
        tet_info = loadmat(file_name, squeeze_me=False)[
            "tetinfo"].squeeze(axis=0)
        tetrode_info.append(
            convert_tetrode_epoch_to_dataframe(
                tet_info[day - 1].squeeze(axis=0)[epoch - 1].squeeze(axis=0),
                epoch_key
            )
        )
        return pd.concat(tetrode_info, sort=True)

    for animal in animals.values():
        file_name = get_tetrode_info_path(animal)
        tet_info = loadmat(file_name, squeeze_me=False)["tetinfo"]
        for day_ind, day in enumerate(tet_info.squeeze(axis=0)):
            for epoch_ind, epoch in enumerate(day.squeeze(axis=0)):
                epoch_key = (
                    animal.short_name,
                    day_ind + 1,
                    epoch_ind + 1,
                )
                tetrode_info.append(
                    convert_tetrode_epoch_to_dataframe(
                        epoch.squeeze(axis=0), epoch_key)
                )

    return pd.concat(tetrode_info, sort=True)


def get_LFP_filename(tetrode_key, animals):
    '''Returns a file name for the tetrode file LFP for an epoch.

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
    filename : str
        File path to tetrode file LFP
    '''
    animal, day, epoch, tetrode_number = tetrode_key
    filename = ('{animal.short_name}eeg{day:02d}-{epoch}-'
                '{tetrode_number:02d}.mat').format(
                    animal=animals[animal], day=day, epoch=epoch,
                    tetrode_number=tetrode_number)
    return join(animals[animal].directory, 'EEG', filename)


def _get_tetrode_id(dataframe):
    '''Unique string identifier for a tetrode'''
    return (dataframe.animal + '_' +
            dataframe.day.map('{:02d}'.format) + '_' +
            dataframe.epoch.map('{:02}'.format) + '_' +
            dataframe.tetrode_number.map('{:03}'.format))


def convert_tetrode_epoch_to_dataframe(tetrodes_in_epoch, epoch_key):
    """Convert tetrode information data structure to dataframe.

    Parameters
    ----------
    tetrodes_in_epoch : matlab data structure
    epoch_key : tuple
        Unique key identifying a recording epoch. Elements are
        (animal, day, epoch)

    Returns
    -------
    tetrode_info : dataframe

    """
    animal, day, epoch = epoch_key
    tetrode_dict_list = [_convert_to_dict(
        tetrode) for tetrode in tetrodes_in_epoch]
    return (pd.DataFrame(tetrode_dict_list)
            .assign(animal=lambda x: animal)
            .assign(day=lambda x: day)
            .assign(epoch=lambda x: epoch)
            .assign(tetrode_number=lambda x: x.index + 1)
            .assign(tetrode_id=_get_tetrode_id)
            .set_index(["animal", "day", "epoch", "tetrode_number"])
            .sort_index()
            )


def get_trial_time(epoch_key, animals):
    """Time in the recording session in terms of the LFP.

    This will return the LFP time of the first tetrode found (according to the
    tetrode info). This is useful when there are slightly different timings
    for the recordings and you need a common time.

    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying a recording epoch with elements
        (animal, day, epoch)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    time : pandas.Index

    """
    tetrode_info = make_tetrode_dataframe(animals, epoch_key=epoch_key)
    for tetrode_key in tetrode_info.index:
        lfp_df = get_LFP_dataframe(tetrode_key, animals)
        if lfp_df is not None:
            break

    return lfp_df.index.rename("time")


def get_LFPs(tetrode_keys, animals):
    '''Retrieves LFP data and makes sure the data has the same timestamps.

    This function is useful when data is collected on different signal
    processing systems and have slightly different timings. This function will
    interpolate the data to the time of the first tetrode recorded.

    Parameters
    ----------
    tetrode_keys : list of tuples, shape (n_signals,)
    animals : dict of namedtuples, shape (n_animals)

    Returns
    -------
    LFPs : pandas DataFrame, shape (n_time, n_signals)

    '''
    epoch_key = tetrode_keys[0][:3]
    time = get_trial_time(epoch_key, animals)
    LFPs = pd.concat([get_LFP_dataframe(tetrode_key, animals)
                      for tetrode_key in tetrode_keys], axis=1)
    new_index = pd.Index(np.unique(np.concatenate(
        (LFPs.index, time))), name='time')
    return (LFPs.reindex(index=new_index)
            .interpolate(method='time')
            .reindex(index=time))
