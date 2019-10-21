import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

from .core import get_data_structure
from .tetrodes import get_trial_time
from .track_segment_classification import (calculate_linear_distance,
                                           classify_track_segments)
from .well_traversal_classification import score_inbound_outbound, segment_path


def _get_pos_dataframe(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure(animals[animal], day, 'pos', 'pos')[epoch - 1]
    position_data = struct['data'][0, 0]
    FIELD_NAMES = ['time', 'x_position', 'y_position', 'head_direction',
                   'speed', 'smoothed_x_position', 'smoothed_y_position',
                   'smoothed_head_direction', 'smoothed_speed']
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')
    n_cols = position_data.shape[1]

    if n_cols > 5:
        # Use the smoothed data if available
        NEW_NAMES = {'smoothed_x_position': 'x_position',
                     'smoothed_y_position': 'y_position',
                     'smoothed_head_direction': 'head_direction',
                     'smoothed_speed': 'speed'}
        return (pd.DataFrame(
            position_data[:, 5:], columns=FIELD_NAMES[5:], index=time)
            .rename(columns=NEW_NAMES))
    else:
        return pd.DataFrame(position_data[:, 1:5], columns=FIELD_NAMES[1:5],
                            index=time)


def get_position_dataframe(epoch_key, animals, use_hmm=True,
                           max_distance_from_well=5,
                           route_euclidean_distance_scaling=1,
                           min_distance_traveled=50,
                           sensor_std_dev=10,
                           spacing=15):
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
    position_df = _get_pos_dataframe(epoch_key, animals)
    if use_hmm:
        position_df = _get_linear_position_hmm(
            epoch_key, animals, position_df,
            max_distance_from_well, route_euclidean_distance_scaling,
            min_distance_traveled, sensor_std_dev,
            spacing=spacing)
    else:
        linear_position_df = _get_linpos_dataframe(
            epoch_key, animals, spacing=spacing)
        position_df = position_df.join(linear_position_df)

    return position_df


def _get_linpos_dataframe(epoch_key, animals, spacing=15):
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
    INCLUDE_FIELDS = ['traj', 'lindist', 'linearVelocity', 'segmentIndex']
    time = pd.TimedeltaIndex(struct['time'][0][0].flatten(), unit='s',
                             name='time')
    new_names = {'time': 'time',
                 'traj': 'labeled_segments',
                 'lindist': 'linear_distance',
                 'linearVelocity': 'linear_velocity',
                 'segmentIndex': 'track_segment_id'}
    data = {new_names[name]: struct[name][0][0][:, 0]
            for name in struct.dtype.names
            if name in INCLUDE_FIELDS}
    position_df = pd.DataFrame(data, index=time)
    SEGMENT_ID_TO_ARM_NAME = {
        1: 'Center Arm',
        2: 'Left Arm',
        3: 'Left Arm',
        4: 'Right Arm',
        5: 'Right Arm',
    }
    position_df = position_df.assign(
        arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
    )
    position_df['linear_position'] = _calulcate_linear_position(position_df)
    position_df['linear_position2'] = _calulcate_linear_position2(
        position_df, spacing=spacing)
    return position_df.assign(linear_speed=np.abs(position_df.linear_velocity))


def calculate_linear_velocity(linear_distance, smooth_duration=0.500,
                              sampling_frequency=29):

    smoothed_linear_distance = gaussian_filter1d(
        linear_distance, smooth_duration * sampling_frequency)

    smoothed_velocity = np.diff(smoothed_linear_distance) * sampling_frequency
    return np.r_[smoothed_velocity[0], smoothed_velocity]


def _calulcate_linear_position(position_df):
    '''Calculate linear distance but map left turns onto the negativie axis and
    right turns onto the positive axis'''
    return (position_df.turn.map({np.nan: np.nan, 'Right': 1, 'Left': -1})
            * position_df.linear_distance)


def _calulcate_linear_position2(position_df, spacing=15):
    '''Calculate linear distance but map the left arm to the
    range(max_linear_distance, max_linear_distance + max_left_arm_distance).'''
    linear_position2 = position_df.linear_distance.copy()

    is_center = (position_df.arm_name == 'Center Arm')

    is_right = (position_df.arm_name == 'Right Arm')
    right_distance = linear_position2[is_right]
    right_distance -= right_distance.min()
    right_distance += linear_position2[is_center].max() + spacing
    linear_position2[is_right] = right_distance

    is_left = (position_df.arm_name == 'Left Arm')
    left_distance = linear_position2[is_left]
    left_distance -= left_distance.min()
    left_distance += linear_position2[is_right].max() + spacing
    linear_position2[is_left] = left_distance
    return linear_position2


def _get_linear_position_hmm(epoch_key, animals, position_df,
                             max_distance_from_well=5,
                             route_euclidean_distance_scaling=1,
                             min_distance_traveled=50,
                             sensor_std_dev=10,
                             spacing=15):
    animal, day, epoch = epoch_key
    struct = get_data_structure(animals[animal], day, 'pos', 'pos')[epoch - 1]
    position_data = struct['data'][0, 0]
    track_graph, center_well_id = make_track_graph(epoch_key, animals)
    position = position_data[:, 1:3]
    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        sensor_std_dev=sensor_std_dev)
    position_df['linear_distance'] = calculate_linear_distance(
        track_graph, track_segment_id, center_well_id, position)
    position_df['track_segment_id'] = track_segment_id
    SEGMENT_ID_TO_ARM_NAME = {0.0: 'Center Arm',
                              1.0: 'Left Arm',
                              2.0: 'Right Arm',
                              3.0: 'Left Arm',
                              4.0: 'Right Arm'}
    position_df = position_df.assign(
        arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
    )

    segments_df, labeled_segments = get_segments_df(
        epoch_key, animals, position_df, max_distance_from_well,
        min_distance_traveled)

    segments_df = pd.merge(
        labeled_segments, segments_df, right_index=True,
        left_on='labeled_segments', how='outer')
    position_df = pd.concat((position_df, segments_df), axis=1)
    position_df['linear_position'] = _calulcate_linear_position(position_df)
    position_df['linear_position2'] = _calulcate_linear_position2(
        position_df, spacing=spacing)
    position_df['linear_velocity'] = calculate_linear_velocity(
        position_df.linear_distance, smooth_duration=0.500,
        sampling_frequency=29)
    position_df['linear_speed'] = np.abs(position_df.linear_velocity)

    return position_df


def get_interpolated_position_dataframe(epoch_key, animals,
                                        time_function=get_trial_time,
                                        use_hmm=True,
                                        max_distance_from_well=5,
                                        route_euclidean_distance_scaling=1,
                                        min_distance_traveled=50,
                                        sensor_std_dev=10,
                                        spacing=15):
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
    max_distance_from_well : float, optional
    route_euclidean_distance_scaling : float, optional
        How much to prefer route distances between successive time points
        that are closer to the euclidean distance. Smaller numbers mean the
        route distance is more likely to be close to the euclidean distance.
    min_distance_traveled : float, optional

    Returns
    -------
    interpolated_position : pandas.DataFrame

    '''
    time = time_function(epoch_key, animals)
    position_df = get_position_dataframe(
        epoch_key, animals, use_hmm, max_distance_from_well,
        route_euclidean_distance_scaling, min_distance_traveled,
        sensor_std_dev, spacing=spacing)
    position_df = position_df.drop(
        ['linear_position', 'linear_position2'], axis=1)

    CONTINUOUS_COLUMNS = ['head_direction', 'speed', 'linear_distance',
                          'x_position', 'y_position',
                          'linear_speed', 'linear_velocity']
    position_categorical = (position_df
                            .drop(CONTINUOUS_COLUMNS, axis=1, errors='ignore')
                            .reindex(index=time, method='pad'))
    position_categorical['is_correct'] = (
        position_categorical.is_correct.fillna(False))

    CATEGORICAL_COLUMNS = ['labeled_segments', 'from_well', 'to_well', 'task',
                           'is_correct', 'turn', 'track_segment_id',
                           'arm_name']
    position_continuous = position_df.drop(CATEGORICAL_COLUMNS, axis=1,
                                           errors='ignore')
    new_index = pd.Index(np.unique(np.concatenate(
        (position_continuous.index, time))), name='time')
    interpolated_position = (position_continuous
                             .reindex(index=new_index)
                             .interpolate(method='time')
                             .reindex(index=time))
    interpolated_position.loc[
        interpolated_position.linear_distance < 0, 'linear_distance'] = 0.0
    interpolated_position.loc[interpolated_position.speed < 0, 'speed'] = 0.0
    interpolated_position.loc[
        interpolated_position.linear_speed < 0, 'linear_speed'] = 0.0

    position_info = position_categorical.join(interpolated_position)

    position_info['linear_position'] = _calulcate_linear_position(
        position_info)
    position_info['linear_position2'] = _calulcate_linear_position2(
        position_info, spacing=spacing)

    return position_info


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


def get_track_segments(epoch_key, animals):
    '''

    Parameters
    ----------
    epoch_key : tuple
    animals : dict of namedtuples

    Returns
    -------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)
    center_well_position : ndarray, shape (n_space,)

    '''
    animal, day, epoch = epoch_key
    task_file = get_data_structure(animals[animal], day, 'task', 'task')
    linearcoord = task_file[epoch - 1]['linearcoord'][0, 0].squeeze()
    track_segments = [np.stack(((arm[:-1, :, 0], arm[1:, :, 0])), axis=1)
                      for arm in linearcoord]
    center_well_position = track_segments[0][0][0]
    track_segments = np.concatenate(track_segments)
    _, unique_ind = np.unique(track_segments, return_index=True, axis=0)
    return track_segments[np.sort(unique_ind)], center_well_position


def make_track_graph(epoch_key, animals):
    '''

    Parameters
    ----------
    epoch_key : tuple, (animal, day, epoch)
    animals : dict of namedtuples

    Returns
    -------
    track_graph : networkx Graph
    center_well_id : int

    '''
    track_segments, center_well_position = get_track_segments(
        epoch_key, animals)
    nodes = track_segments.copy().reshape((-1, 2))
    _, unique_ind = np.unique(nodes, return_index=True, axis=0)
    nodes = nodes[np.sort(unique_ind)]

    edges = np.zeros(track_segments.shape[:2], dtype=np.int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id

    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(), axis=1)

    track_graph = nx.Graph()

    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))

    for edge, distance in zip(edges, edge_distances):
        nx.add_path(track_graph, edge, distance=distance)

    center_well_id = np.unique(
        np.nonzero(np.isin(nodes, center_well_position).sum(axis=1) > 1)[0])[0]

    return track_graph, center_well_id


def get_segments_df(epoch_key, animals, position_df, max_distance_from_well=5,
                    min_distance_traveled=50):
    well_locations = get_well_locations(epoch_key, animals)
    position = position_df.loc[:, ['x_position', 'y_position']].values
    segments_df, labeled_segments = segment_path(
        position_df.index, position, well_locations, epoch_key, animals,
        max_distance_from_well=max_distance_from_well)
    segments_df = score_inbound_outbound(
        segments_df, epoch_key, animals, min_distance_traveled)
    segments_df = segments_df.loc[
        :, ['from_well', 'to_well', 'task', 'is_correct', 'turn']]

    return segments_df, labeled_segments
