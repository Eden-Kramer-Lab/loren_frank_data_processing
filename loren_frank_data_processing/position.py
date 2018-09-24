import numpy as np
import pandas as pd

import networkx as nx

from .core import get_data_structure
from .tetrodes import get_trial_time
from .track_segment_classification import (calculate_linear_distance,
                                           classify_track_segments)
from .well_traversal_classification import score_inbound_outbound, segment_path


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
    FIELD_NAMES = ['time', 'x_position', 'y_position', 'head_direction',
                   'speed', 'smoothed_x_position', 'smoothed_y_position',
                   'smoothed_head_direction', 'smoothed_speed']
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')
    n_cols = position_data.shape[1]

    if n_cols > 5:
        # Use the smoothed data
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
                                        max_distance_from_well=5,
                                        route_euclidean_distance_scaling=1,
                                        min_distance_traveled=50):
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
    min_distance_traveled : float, optional

    Returns
    -------
    interpolated_position : pandas.DataFrame

    '''
    time = time_function(epoch_key, animals)
    position_df = get_position_dataframe(epoch_key, animals)

    track_graph, center_well_id = make_track_graph(epoch_key, animals)
    position = position_df.loc[:, ['x_position', 'y_position']].values
    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling)
    position_df['linear_distance'] = calculate_linear_distance(
        track_graph, track_segment_id, center_well_id, position)

    segments_df, labeled_segments = get_segments_df(
        epoch_key, animals, max_distance_from_well, min_distance_traveled)

    segments_df = pd.merge(
        labeled_segments, segments_df, right_index=True,
        left_on='labeled_segments', how='outer')
    position_df = pd.concat((position_df, segments_df), axis=1)
    position_df['linear_position'] = (
        position_df.turn.map({np.nan: np.nan, 'Right': 1, 'Left': -1})
        * position_df.linear_distance)

    categorical_columns = ['labeled_segments', 'from_well', 'to_well', 'task',
                           'is_correct', 'turn']
    continuous_columns = ['head_direction', 'speed', 'linear_distance',
                          'x_position', 'y_position', 'linear_position']
    position_categorical = (position_df
                            .drop(continuous_columns, axis=1)
                            .reindex(index=time, method='pad'))
    position_categorical['is_correct'] = (
        position_categorical.is_correct.fillna(False))
    position_continuous = position_df.drop(categorical_columns, axis=1)
    new_index = pd.Index(np.unique(np.concatenate(
        (position_continuous.index, time))), name='time')
    interpolated_position = (position_continuous
                             .reindex(index=new_index)
                             .interpolate(method='time')
                             .reindex(index=time))
    interpolated_position.loc[
        interpolated_position.linear_distance < 0, 'linear_distance'] = 0
    interpolated_position.loc[interpolated_position.speed < 0, 'speed'] = 0

    return pd.concat([position_categorical, interpolated_position], axis=1)


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
    return (np.unique(np.concatenate(track_segments), axis=0),
            center_well_position)


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
    nodes = np.unique(track_segments.reshape((-1, 2)), axis=0)

    edges = np.zeros(track_segments.shape[:2], dtype=int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id

    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(), axis=1)

    track_graph = nx.Graph()

    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))

    for edge, distance in zip(edges, edge_distances):
        track_graph.add_edge(edge[0], edge[1], distance=distance)

    center_well_id = np.unique(
        np.nonzero(np.isin(nodes, center_well_position).sum(axis=1) > 1)[0])[0]

    return track_graph, center_well_id


def get_segments_df(epoch_key, animals, max_distance_from_well=5,
                    min_distance_traveled=50):
    well_locations = get_well_locations(epoch_key, animals)
    position_df = get_position_dataframe(epoch_key, animals)
    position = position_df.loc[:, ['x_position', 'y_position']].values
    segments_df, labeled_segments = segment_path(
        position_df.index, position, well_locations, epoch_key, animals,
        max_distance_from_well=max_distance_from_well)
    segments_df = score_inbound_outbound(
        segments_df, epoch_key, animals, min_distance_traveled)
    segments_df = segments_df.loc[
            :, ['from_well', 'to_well', 'task', 'is_correct', 'turn']]

    return segments_df, labeled_segments
