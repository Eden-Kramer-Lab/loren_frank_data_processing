import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

from .core import get_data_structure
from .tetrodes import get_trial_time
from .track_segment_classification import (calculate_linear_distance,
                                           classify_track_segments)
from .well_traversal_classification import score_inbound_outbound, segment_path

EDGE_ORDER = [0, 2, 4, 1, 3]
EDGE_SPACING = [15, 0, 15, 0]


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
                           sensor_std_dev=5,
                           diagonal_bias=1E-1,
                           edge_spacing=EDGE_SPACING,
                           edge_order=EDGE_ORDER,
                           skip_linearization=False):
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
    if not skip_linearization:
        if use_hmm:
            position_df = _get_linear_position_hmm(
                epoch_key, animals, position_df,
                max_distance_from_well, route_euclidean_distance_scaling,
                min_distance_traveled, sensor_std_dev, diagonal_bias,
                edge_order=edge_order, edge_spacing=edge_spacing)
        else:
            linear_position_df = _get_linpos_dataframe(
                epoch_key, animals, edge_order=edge_order,
                edge_spacing=edge_spacing)
            position_df = position_df.join(linear_position_df)

    return position_df


def _get_linpos_dataframe(epoch_key, animals, edge_spacing=EDGE_SPACING,
                          edge_order=EDGE_ORDER):
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
    track_graph, center_well_id = make_track_graph(epoch_key, animals)
    position_df['linear_position'] = _calulcate_linear_position(
        position_df.linear_distance.values,
        position_df.track_segment_id.values, track_graph, center_well_id,
        edge_order=edge_order, edge_spacing=edge_spacing)
    return position_df.assign(linear_speed=np.abs(position_df.linear_velocity))


def calculate_linear_velocity(linear_distance, smooth_duration=0.500,
                              sampling_frequency=29):

    smoothed_linear_distance = gaussian_filter1d(
        linear_distance, smooth_duration * sampling_frequency)

    smoothed_velocity = np.diff(smoothed_linear_distance) * sampling_frequency
    return np.r_[smoothed_velocity[0], smoothed_velocity]


def convert_linear_distance_to_linear_position(
        linear_distance, edge_id, edge_order, spacing=30):
    linear_position = linear_distance.copy()
    n_edges = len(edge_order)
    if isinstance(spacing, int) | isinstance(spacing, float):
        spacing = [spacing, ] * (n_edges - 1)

    for prev_edge, cur_edge, space in zip(
            edge_order[:-1], edge_order[1:], spacing):
        is_cur_edge = (edge_id == cur_edge)
        is_prev_edge = (edge_id == prev_edge)

        cur_distance = linear_position[is_cur_edge]
        cur_distance -= cur_distance.min()
        cur_distance += linear_position[is_prev_edge].max() + space
        linear_position[is_cur_edge] = cur_distance

    return linear_position


def get_graph_1D_2D_relationships(track_graph, edge_order, edge_spacing,
                                  center_well_id):
    '''

    Parameters
    ----------
    track_graph : networkx.Graph
    edge_order : array-like, shape (n_edges,)
    edge_spacing : float or array-like, shape (n_edges,)
    center_well_id : int

    Returns
    -------
    node_linear_position : numpy.ndarray, shape (n_edges, n_position_dims)
    edges : numpy.ndarray, shape (n_edges, 2)
    node_2D_position : numpy.ndarray, shape (n_edges, 2, n_position_dims)
    edge_dist : numpy.ndarray, shape (n_edges,)

    '''
    linear_distance = []
    edge_id = []

    dist = dict(
        nx.all_pairs_dijkstra_path_length(track_graph, weight="distance")
    )
    n_edges = len(track_graph.edges)

    for ind, (node1, node2) in enumerate(track_graph.edges):
        linear_distance.append(dist[center_well_id][node1])
        linear_distance.append(dist[center_well_id][node2])
        edge_id.append(ind)
        edge_id.append(ind)

    linear_distance = np.array(linear_distance)
    edge_id = np.array(edge_id)

    node_linear_position = convert_linear_distance_to_linear_position(
        linear_distance, edge_id, edge_order, spacing=edge_spacing
    )

    node_linear_position = node_linear_position.reshape((n_edges, 2))[
        edge_order]
    node_linear_distance = linear_distance.reshape((n_edges, 2))[edge_order]

    return node_linear_position, node_linear_distance


def _calulcate_linear_position(linear_distance, edge_id, track_graph,
                               center_well_id, edge_order, edge_spacing=15):
    '''Calculate linear distance but map the left arm to the
    range(max_linear_distance, max_linear_distance + max_left_arm_distance).'''
    linear_position = linear_distance.copy()

    node_linear_position, node_linear_distance = get_graph_1D_2D_relationships(
        track_graph, edge_order, edge_spacing, center_well_id)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    for start_linear_position, start_linear_distance, cur_edge in zip(
            node_linear_position[:, 0], node_linear_distance[:, 0],
            edge_order):
        is_cur_edge = (edge_id == cur_edge)

        cur_distance = linear_distance[is_cur_edge] - start_linear_distance
        cur_distance += start_linear_position
        linear_position[is_cur_edge] = cur_distance

    return linear_position


def _get_linear_position_hmm(epoch_key, animals, position_df,
                             max_distance_from_well=5,
                             route_euclidean_distance_scaling=1,
                             min_distance_traveled=50,
                             sensor_std_dev=5,
                             diagonal_bias=1E-1,
                             edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING,
                             position_sampling_frequency=33):
    animal, day, epoch = epoch_key
    track_graph, center_well_id = make_track_graph(epoch_key, animals)
    position = position_df.loc[:, ['x_position', 'y_position']].values
    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        sensor_std_dev=sensor_std_dev,
        diagonal_bias=diagonal_bias)
    (position_df['linear_distance'],
     position_df['projected_x_position'],
     position_df['projected_y_position']) = calculate_linear_distance(
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
    position_df['linear_position'] = _calulcate_linear_position(
        position_df.linear_distance.values,
        position_df.track_segment_id.values, track_graph, center_well_id,
        edge_order=edge_order, edge_spacing=edge_spacing)
    position_df['linear_velocity'] = calculate_linear_velocity(
        position_df.linear_distance, smooth_duration=0.500,
        sampling_frequency=position_sampling_frequency)
    position_df['linear_speed'] = np.abs(position_df.linear_velocity)
    position_df['is_correct'] = position_df.is_correct.fillna(False)

    return position_df


def get_interpolated_position_dataframe(epoch_key, animals,
                                        time_function=get_trial_time,
                                        use_hmm=True,
                                        max_distance_from_well=5,
                                        route_euclidean_distance_scaling=1,
                                        min_distance_traveled=50,
                                        sensor_std_dev=5,
                                        diagonal_bias=1E-1,
                                        edge_spacing=EDGE_SPACING,
                                        edge_order=EDGE_ORDER,
                                        position_sampling_frequency=1500):
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
        This favors less jumps. Larger numbers favor more jumps.
    min_distance_traveled : float, optional

    Returns
    -------
    interpolated_position : pandas.DataFrame

    '''
    time = time_function(epoch_key, animals)
    position_df = get_position_dataframe(
        epoch_key, animals, skip_linearization=True)

    new_index = pd.Index(np.unique(np.concatenate(
        (position_df.index, time))), name='time')
    position_df = (position_df
                   .reindex(index=new_index)
                   .interpolate(method='linear')
                   .reindex(index=time))

    position_df.loc[position_df.speed < 0, 'speed'] = 0.0

    position_df = _get_linear_position_hmm(
        epoch_key, animals, position_df,
        max_distance_from_well, route_euclidean_distance_scaling,
        min_distance_traveled, sensor_std_dev, diagonal_bias,
        edge_order=edge_order, edge_spacing=edge_spacing,
        position_sampling_frequency=position_sampling_frequency)

    return position_df


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
    linearcoord = task_file[epoch - 1]['linearcoord'][0, 0].squeeze(axis=0)
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
        np.diff(track_segments, axis=-2).squeeze(axis=-2), axis=1)

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
