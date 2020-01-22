import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, plot_track, project_points_to_segment)


def _get_projected_track_position(track_graph, track_segment_id, position):
    track_segment_id[np.isnan(track_segment_id)] = 0
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_position = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_position.shape[0]
    return projected_track_position[(
        np.arange(n_time), track_segment_id)]


def make_actual_vs_linearized_position_movie(
        track_graph, position_df, time_slice=None,
        movie_name='actual_vs_linearized', frame_rate=33):
    '''

    Parameters
    ----------
    track_graph : networkx.Graph
    position_df : pandas.DataFrame
    time_slice : slice or None, optional
    movie_name : str, optional
    frame_rate : float, optional
        Frames per second.
    '''

    all_position = position_df.loc[:, ['x_position', 'y_position']].values
    all_linear_position = position_df.linear_position.values
    all_time = position_df.index.values / np.timedelta64(1, 's')

    if time_slice is None:
        position = all_position
        track_segment_id = position_df.track_segment_id.values
        linear_position = all_linear_position
        time = all_time
    else:
        position = all_position[time_slice]
        track_segment_id = position_df.iloc[time_slice].track_segment_id.values
        linear_position = all_linear_position[time_slice]
        time = all_time[time_slice]

    projected_track_position = _get_projected_track_position(
        track_graph, track_segment_id, position)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig, axes = plt.subplots(1, 2, figsize=(21, 7), constrained_layout=True,
                             gridspec_kw={'width_ratios': [2, 1]})

    # Subplot 1
    axes[0].scatter(all_time, all_linear_position, color='lightgrey',
                    zorder=0, s=10)
    axes[0].set_xlim((all_time.min(), all_time.max()))
    axes[0].set_ylim((all_linear_position.min(), all_linear_position.max()))
    linear_head = axes[0].scatter([], [], s=100, zorder=101, color='b')

    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Position [cm]')
    axes[0].set_title('Linearized Position')

    axes[1].plot(all_position[:, 0], all_position[:, 1], color='lightgrey',
                 zorder=-10)
    plot_track(track_graph, ax=axes[1])
    plt.axis('off')

    # Subplot 2
    axes[1].set_xlim(all_position[:, 0].min() - 10,
                     all_position[:, 0].max() + 10)
    axes[1].set_ylim(all_position[:, 1].min() - 10,
                     all_position[:, 1].max() + 10)

    actual_line, = axes[1].plot(
        [], [], 'g-', label='actual position', linewidth=3, zorder=101)
    actual_head = axes[1].scatter([], [], s=80, zorder=101, color='g')

    predicted_line, = axes[1].plot(
        [], [], 'b-', label='linearized position', linewidth=3, zorder=102)
    predicted_head = axes[1].scatter([], [], s=80, zorder=102, color='b')

    axes[1].legend()
    axes[1].set_xlabel('x-position')
    axes[1].set_ylabel('y-position')
    axes[1].set_title('Linearized vs. Actual Position')

    def _update_plot(time_ind):
        start_ind = max(0, time_ind - 33)
        time_slice = slice(start_ind, time_ind)

        linear_head.set_offsets(
            np.array((time[time_ind], linear_position[time_ind])))

        actual_line.set_data(position[time_slice, 0], position[time_slice, 1])
        actual_head.set_offsets(position[time_ind])

        predicted_line.set_data(projected_track_position[time_slice, 0],
                                projected_track_position[time_slice, 1])
        predicted_head.set_offsets(projected_track_position[time_ind])

        return actual_line, predicted_line

    n_time = position.shape[0]
    line_ani = animation.FuncAnimation(fig, _update_plot, frames=n_time,
                                       interval=1000 / frame_rate, blit=True)
    line_ani.save(movie_name + '.mp4', writer=writer)
