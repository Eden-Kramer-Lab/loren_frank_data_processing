import re
from glob import glob
from itertools import chain
from os import listdir, makedirs, walk
from os.path import join
from shutil import copyfile

import numpy as np
import pandas as pd


def copy_animal(animal, src_directory):
    '''Copies essential data files and renames multiunit files.

    Parameters
    ----------
    animal : namedtuple
        First element is the target directory where the animal's data
        should be located. The second element is the animal shortened name.
    src_directory : str

    '''
    processed_data_dir = join(src_directory, animal.short_name)
    try:
        makedirs(animal.directory)
    except FileExistsError:
        pass

    FILE_TYPES = ['cellinfo', 'linpos', 'pos', 'rawpos', 'task', 'tetinfo',
                  'spikes']
    data_files = [glob(join(processed_data_dir,
                            '{animal.short_name}{file_type}*.mat').format(
        animal=animal, file_type=file_type))
        for file_type in FILE_TYPES]
    for old_path in chain.from_iterable(data_files):
        new_path = join(animal.directory, old_path.split('/')[-1])

        print('Copying {old_path}\nto \n{new_path}\n'.format(
            old_path=old_path,
            new_path=new_path
        ))
        copyfile(old_path, new_path)

    src_lfp_data_dir = join(processed_data_dir, 'EEG')
    target_lfp_data_dir = join(animal.directory, 'EEG')
    try:
        makedirs(target_lfp_data_dir)
    except FileExistsError:
        pass
    lfp_files = [file for file in listdir(src_lfp_data_dir)
                 if 'gnd' not in file and 'eeg' in file]

    for file_name in lfp_files:
        old_path = join(src_lfp_data_dir, file_name)
        new_path = join(target_lfp_data_dir, file_name)

        print('Copying {old_path}\nto \n{new_path}\n'.format(
            old_path=old_path,
            new_path=new_path
        ))
        copyfile(old_path, new_path)

    marks_directory = join(src_directory, animal.directory)
    mark_files = [join(root, f) for root, _, files in walk(marks_directory)
                  for f in files if f.endswith('_params.mat')
                  and not f.startswith('matclust')]
    new_mark_filenames = [rename_mark_file(mark_file, animal)
                          for mark_file in mark_files]
    for mark_file, new_filename in zip(mark_files, new_mark_filenames):
        mark_path = join(target_lfp_data_dir, new_filename)
        print('Copying {mark_file}\nto \n{new_filename}\n'.format(
            mark_file=mark_file,
            new_filename=mark_path
        ))
        copyfile(mark_file, mark_path)


def rename_mark_file(file_str, animal):
    matched = re.match(
        r'.*(\d.+)-(\d.+)_params.mat', file_str.split('/')[-1])
    try:
        day, tetrode_number = matched.groups()
    except AttributeError:
        matched = re.match(
            r'.*(\d+)-.*-(\d+)_params.mat', file_str.split('/')[-1])
        try:
            day, tetrode_number = matched.groups()
        except AttributeError:
            print(file_str)
            raise

    new_name = ('{animal}marks{day:02d}-{tetrode_number:02d}.mat'.format(
        animal=animal.short_name,
        day=int(day),
        tetrode_number=int(tetrode_number)
    ))

    return new_name


def find_closest_ind(search_array, target):
    '''Finds the index position in the search_array that is closest to the
    target. This works for large search_arrays. See:
    http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python

    Parameters
    ----------
    search_array : ndarray
    target : ndarray element

    Returns
    -------
    index : int
        Index closest to target element.

    '''
    # Get insertion index, need to be sorted
    ind = search_array.searchsorted(target)
    # If index is out of bounds, move to bounds
    ind = np.clip(ind, 1, len(search_array) - 1)
    # Adjust if the left or right index is closer
    adjust = (target - search_array[ind - 1]
              ) < (search_array[ind] - target)
    return ind - adjust


def _get_windowed_dataframe(time_series, segments, window_start, window_end):
    '''For each segment, return a dataframe with the time relative to
    the start of the segment + window_offset.
    '''
    for segment_start, segment_end in segments:
        if window_end is not None:
            segment_end = segment_start + window_end
        time_series_segment = time_series.loc[
            (segment_start + window_start):segment_end, :]
        new_time = time_series_segment.index - segment_start
        yield time_series_segment.set_index(new_time)


def reshape_to_segments(time_series, segments, window_offset=None, axis=0):
    '''Take multiple windows of a time series and set time relative to
    the start of the window.

    Useful for examining an event of interest.

    Parameters
    ----------
    time_series : pandas.DataFrame, shape (n_time, ...)
        Time series to be segmented. Index of time series must be the time
        of the time series and be named `time`.
    segments : pandas.DataFrame, shape (n_segments, 2)
        Start and end time for each time segment. Each column corresponds to
        segment_start and segment_end.
    window_offset : None or tuple (window_start, window_end), optional
        Offset the window relative to the start time of the segment. If
        `window_offset` is None, the returned segment will be the time series
        slice (window_start, window_end). If `window_offset` is given, then the
        returned window will be (window_start + segment_start, window_end). If
        window_start or window_end is None, then it will be replaced with
        segment_start and segment_end, respectivity
    axis : int, optional
        Concat axis

    Returns
    -------
    segmented_time_series : pandas DataFrame

    Examples
    --------
    >>> n_time = 10
    >>> time = pd.Index(np.arange(0, n_time) / 1000, name='time')
    >>> time_series = pd.DataFrame({'data': np.arange(n_time)}, index=time)
    >>> reshape_to_segments(time_series, [(0.001, 0.004), (0.006, 0.008)])
    >>> reshape_to_segments(time_series, [(0.001, 0.004), (0.006, 0.008)],
                            window_offset=(-0.001, None))
    >>> reshape_to_segments(time_series, [(0.001, 0.004), (0.006, 0.008)],
                            window_offset=(-0.001, 0.001))

    '''
    segments_index = segments.index
    segments = segments.itertuples(index=False)
    time_series = pd.DataFrame(time_series)

    if window_offset is not None:
        window_start, window_end = window_offset
        if window_start is not None:
            window_start = pd.Timedelta(seconds=window_start)
        else:
            window_start = pd.Timedelta(seconds=0)
        if window_end is not None:
            window_end = pd.Timedelta(seconds=window_end)
    else:
        window_start, window_end = pd.Timedelta(seconds=0), None

    return (pd.concat(_get_windowed_dataframe(
                time_series, segments, window_start, window_end),
            keys=segments_index,
            axis=axis).sort_index())
