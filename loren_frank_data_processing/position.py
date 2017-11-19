import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label

from .core import get_data_structure
from .tetrodes import get_trial_time


def get_position_dataframe(epoch_key, animals):
    '''Returns a list of position dataframes with a length corresponding
     to the number of epochs in the epoch key -- either a tuple or a
    list of tuples with the format (animal, day, epoch_number)

    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying a recording epoch. Elements are
        (animal, day, epoch)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    position : pandas dataframe
        Contains information about the animal's position, head direction,
        and speed.

    '''
    animal, day, epoch = epoch_key
    struct = get_data_structure(animals[animal], day, 'pos', 'pos')[epoch - 1]
    position_data = struct['data'][0, 0]
    field_names = struct['fields'][0, 0].item().split()
    NEW_NAMES = {'x': 'x_position',
                 'y': 'y_position',
                 'dir': 'head_direction',
                 'vel': 'speed'}
    time = pd.TimedeltaIndex(
        position_data[:, field_names.index('time')], unit='s', name='time')
    return (pd.DataFrame(
        position_data, columns=field_names, index=time)
        .rename(columns=NEW_NAMES)
        .drop([name for name in field_names
               if name not in NEW_NAMES], axis=1))


def get_linear_position_structure(epoch_key, animals):
    '''The time series of linearized (1D) positions of the animal for a given
    epoch.

    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying a recording epoch. Elements are
        (animal, day, epoch)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    linear_position : pandas.DataFrame

    '''
    animal, day, epoch = epoch_key
    struct = get_data_structure(
        animals[animal], day, 'linpos', 'linpos')[epoch - 1][0][0][
            'statematrix']
    INCLUDE_FIELDS = ['traj', 'lindist']
    time = pd.TimedeltaIndex(struct['time'][0][0].flatten(), unit='s',
                             name='time')
    new_names = {'time': 'time', 'traj': 'trajectory_category_ind',
                 'lindist': 'linear_distance'}
    data = {new_names[name]: struct[name][0][0].flatten()
            for name in struct.dtype.names
            if name in INCLUDE_FIELDS}
    return pd.DataFrame(data, index=time)


def get_interpolated_position_dataframe(epoch_key, animals,
                                        time_function=get_trial_time,
                                        max_distance_from_well=15):
    '''Gives the interpolated position of animal for a given epoch.

    Defaults to interpolating the position to the LFP time. Can use the
    `time_function` to specify different time to interpolate to.

    Parameters
    ----------
    epoch_key : tuple
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    time_function : function, optional
        Function that take an epoch key (animal_short_name, day, epoch) that
        defines the time the multiunits are relative to. Defaults to using
        the time the LFPs are sampled at.

    Returns
    -------
    interpolated_position : pandas.DataFrame

    '''
    time = time_function(epoch_key, animals)
    position = (pd.concat(
        [get_linear_position_structure(epoch_key, animals),
         get_position_dataframe(epoch_key, animals)], axis=1)
         .drop('trajectory_category_ind', axis=1)
    )
    old_dt = (position.index[1] - position.index[0]).total_seconds()

    well_locations = get_well_locations(epoch_key, animals)
    xy = np.stack((position.x_position, position.y_position), axis=1)
    segments_df, labeled_segments = segment_path(
        position.index, xy, well_locations,
        max_distance_from_well=max_distance_from_well)
    segments_df = score_inbound_outbound(segments_df).loc[
        :, ['from_well', 'to_well', 'task', 'is_correct']]

    segments_df = pd.merge(
        labeled_segments, segments_df, right_index=True,
        left_on='labeled_segments', how='outer')
    position = pd.concat((position, segments_df), axis=1)

    categorical_columns = ['labeled_segments', 'from_well', 'to_well', 'task',
                           'is_correct']
    continuous_columns = ['head_direction', 'speed', 'linear_distance',
                          'x_position', 'y_position']
    position_categorical = (position
                            .drop(continuous_columns, axis=1)
                            .reindex(index=time, method='pad'))
    position_continuous = position.drop(categorical_columns, axis=1)
    new_index = pd.Index(np.unique(np.concatenate(
        (position_continuous.index, time))), name='time')
    interpolated_position = (position_continuous
                             .reindex(index=new_index)
                             .interpolate(method='values')
                             .reindex(index=time))
    interpolated_position.loc[
        interpolated_position.linear_distance < 0, 'linear_distance'] = 0
    interpolated_position.loc[interpolated_position.speed < 0, 'speed'] = 0
    limit = np.ceil(old_dt / (time[1] - time[0]).total_seconds()).astype(int)
    return (pd.concat([position_categorical, interpolated_position], axis=1)
            .fillna(method='backfill', limit=limit))


def paired_distances(x, y):
    '''Euclidean distance between x and y at each time point.

    Parameters
    ----------
    x,y : ndarray, shape (n_time, n_space)

    Returns
    -------
    distance : ndarray, shape (n_time,)

    '''
    x, y = np.array(x), np.array(y)
    x = np.atleast_2d(x).T if x.ndim < 2 else x
    y = np.atleast_2d(y).T if y.ndim < 2 else y
    return np.linalg.norm(x - y, axis=1)


def enter_exit_target(position, target, max_distance=1):
    '''Marks when a position has reached a target ("enter")
    and when it has left a target ("exit").

    The position is considered to have reached a target when it is less than
    the `max_distance` from the target.

    Enter and exit times are marked as follows:
     1: entered the target radius
     0: neither
    -1: exited the target radius

    Works for 1D position and 2D position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_space)
    target : ndarray, shape (1, n_space)
    max_distance : float, optional
        How close the position is to the target to be considered at the target.

    Returns
    -------
    enter_exit : ndarray, shape (n_time,)
    at_target : ndarray, shape (n_time,)

    '''
    distance_from_target = paired_distances(position, target)
    at_target = distance_from_target < max_distance
    enter_exit = np.r_[0, np.diff(at_target.astype(float))]
    return enter_exit, at_target


def shift_well_enters(enter_exit):
    '''Shifts the enter times back one time point.'''
    shifted_enter_exit = enter_exit.copy()
    old_ind = np.where(enter_exit > 0)[0]  # positive entries are well-entries
    new_ind = old_ind - 1
    shifted_enter_exit[new_ind] = enter_exit[old_ind]
    shifted_enter_exit[old_ind] = 0
    return shifted_enter_exit


def segment_path(time, position, well_locations, max_distance_from_well=10):
    '''Label traversals between each well location.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, n_space)
    well_locations : array_like, shape (n_wells, n_space)
    max_distance_from_well : float, optional
        The animal is considered at a well location if its position is closer
        than this value.

    Returns
    -------
    segments_df : pandas DataFrame, shape (n_segments, 6)
    labeled_segments : pandas DataFrame, shape (n_time,)

    '''
    n_wells = len(well_locations)
    well_enter_exit, at_target = np.stack(
        [enter_exit_target(position, np.atleast_2d(well),
                           max_distance_from_well)
         for well in well_locations], axis=1)

    well_labels = np.arange(n_wells) + 1
    well_enter_exit = np.sum(well_enter_exit.T * well_labels, axis=1)
    shifted_well_enter_exit = shift_well_enters(well_enter_exit)
    is_segment = ~(np.sum(at_target, axis=0) > 0)
    labeled_segments, n_segment_labels = label(is_segment)
    segment_labels = np.arange(n_segment_labels) + 1

    start_time, end_time, duration = [], [], []
    distance_traveled, from_well, to_well = [], [], []

    for segment_label in segment_labels:
        is_seg = np.in1d(labeled_segments, segment_label)
        segment_time = time[is_seg]
        start_time.append(segment_time.min())
        end_time.append(segment_time.max())
        duration.append(segment_time.max() - segment_time.min())
        try:
            start, _, end = np.unique(shifted_well_enter_exit[is_seg])
        except ValueError:
            start, end = np.nan, np.nan

        from_well.append(np.abs(start))
        to_well.append(np.abs(end))
        p = position[is_seg]
        distance_traveled.append(np.sum(paired_distances(p[1:], p[:-1])))

    data = [('start_time', start_time), ('end_time', end_time),
            ('duration', duration), ('from_well', from_well),
            ('to_well', to_well),
            ('distance_traveled', distance_traveled)]
    index = pd.Index(segment_labels, name='segment')
    return (pd.DataFrame.from_items(data).set_index(index),
            pd.DataFrame(dict(labeled_segments=labeled_segments), index=time))


def find_last_non_center_well(segments_df, segment_ind):
    last_wells = segments_df.iloc[:segment_ind].to_well
    try:
        return last_wells[last_wells != 'center'].iloc[-1]
    except IndexError:
        # There are no non-center wells. Just return current well.
        return ''


def get_correct_inbound_outbound(segments_df):
    n_segments = segments_df.shape[0]
    task = np.empty((n_segments,), dtype=object)
    is_correct = np.empty((n_segments,), dtype=bool)

    for segment_ind in np.arange(n_segments):
        if segments_df.iloc[segment_ind].from_well == 'center':
            task[segment_ind] = 'Outbound'
            is_correct[segment_ind] = (
                segments_df.iloc[segment_ind].to_well !=
                find_last_non_center_well(segments_df, segment_ind))
        else:
            task[segment_ind] = 'Inbound'
            is_correct[segment_ind] = (
                segments_df.iloc[segment_ind].to_well == 'center')

    segments_df['task'] = task
    segments_df['is_correct'] = is_correct

    return segments_df


def score_inbound_outbound(segments_df, min_distance_traveled=100):
    '''In the alternating arm task, determines whether the trial should be
    inbound (running to the center arm) or outbound (running to the opposite
    outer arm as before) and if the trial was performed correctly.

    Parameters
    ----------
    segments_df : pandas DataFrame
        Output of `segment_path` function.
    min_distance_traveled : float, optional
        Minimum path length (in cm) while outside of the well radius for
        a segment to be considered as a trial.

    Returns
    -------
    segments_df : pandas DataFrame
        Same as the input dataframe but with the wells labeled
        (left, right, center) and columns for `task` (inbound/outbound) and
        `is_correct` (True/False).
    '''
    segments_df = (segments_df.copy()
                   .loc[segments_df.distance_traveled > min_distance_traveled]
                   .dropna())
    WELL_NAMES = {
        1: 'center',
        2: 'left',
        3: 'right'
    }
    segments_df = segments_df.assign(
        to_well=lambda df: df.to_well.map(WELL_NAMES),
        from_well=lambda df: df.from_well.map(WELL_NAMES))
    return get_correct_inbound_outbound(segments_df)


def get_well_locations(epoch_key, animals):
    '''Retrieves the 2D coordinates for each well.
    '''
    animal, day, epoch = epoch_key
    task_file = get_data_structure(animals[animal], day, 'task', 'task')
    linearcoord = task_file[epoch - 1]['linearcoord'][0, 0].squeeze()
    well_locations = []
    for arm in linearcoord:
        well_locations.append(arm[0, :, 0])
        well_locations.append(arm[-1, :, 0])
    well_locations = np.stack(well_locations)
    _, ind = np.unique(well_locations, axis=0, return_index=True)
    return well_locations[np.sort(ind), :]
