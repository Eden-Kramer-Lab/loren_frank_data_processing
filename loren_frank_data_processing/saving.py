from glob import glob
from os.path import isfile, join
from warnings import filterwarnings

import pandas as pd
from xarray.backends.api import (_CONCAT_DIM_DEFAULT, _default_lock,
                                 _MultiFileCloser, auto_combine, basestring,
                                 open_dataset)


def get_analysis_file_path(processed_data_dir, animal, day, epoch):
    '''File path for analysis file.
    '''
    filename = '{animal}_{day:02d}_{epoch:02d}.nc'.format(
        animal=animal, day=day, epoch=epoch)
    return join(processed_data_dir, filename)


def save_xarray(processed_data_dir, epoch_key, dataset, group=''):
    '''Saves xarray data to file corresponding to epoch key

    Parameters
    ----------
    processed_data_dir : str
        Path to processed data directory
    epoch_key : tuple
        (Animal, day, epoch)
    dataset : xarray Dataset or DataArray
        Data to be saved
    group : str, optional
        HDF5 group name

    '''
    path = get_analysis_file_path(processed_data_dir, *epoch_key)
    write_mode = 'a' if isfile(path) else 'w'
    dataset.to_netcdf(path=path, group=group, mode=write_mode)


def _open_dataset(*args, **kwargs):
    try:
        return open_dataset(*args, **kwargs)
    except (IndexError, OSError):
        return None


def open_mfdataset(paths, chunks=None, concat_dim=_CONCAT_DIM_DEFAULT,
                   compat='no_conflicts', preprocess=None, engine=None,
                   lock=None, **kwargs):
    '''Open multiple files as a single dataset.

    This function is adapted from the xarray function of the same name.
    The main difference is that instead of failing on files that do not
    exist, this function keeps processing.

    Requires dask to be installed.  Attributes from the first dataset file
    are used for the combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an
        explicit list of files to open.
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by
        chunk sizes. In general, these should divide the dimensions of each
        dataset. If int, chunk each dimension by ``chunks``.
        By default, chunks will be chosen to load entire input files into
        memory at once. This has a major impact on performance: please see
        the full documentation for more details.
    concat_dim : None, str, DataArray or Index, optional
        Dimension to concatenate files along. This argument is passed on to
        :py:func:`xarray.auto_combine` along with the dataset objects. You
        only need to provide this argument if the dimension along which you
        want to concatenate is not a dimension in the original datasets,
        e.g., if you want to stack a collection of 2D arrays along a third
        dimension. By default, xarray attempts to infer this argument by
        examining component files. Set ``concat_dim=None`` explicitly to
        disable concatenation.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:
        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
    preprocess : callable, optional
        If provided, call this function on each dataset prior to
        concatenation.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default
        engine is chosen based on available dependencies, with a preference
        for 'netcdf4'.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many
        files being open.  However, this option doesn't work with streams,
        e.g., BytesIO.
    lock : False, True or threading.Lock, optional
        This argument is passed on to :py:func:`dask.array.from_array`. By
        default, a per-variable lock is used when reading data from netCDF
        files with the netcdf4 and h5netcdf engines to avoid issues with
        concurrent access when using dask's multithreaded backend.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.
    Returns
    -------
    xarray.Dataset
    See Also
    --------
    auto_combine
    open_dataset

    '''
    filterwarnings('ignore', 'elementwise comparison failed;')
    filterwarnings('ignore', 'numpy equal will not check object')

    if isinstance(paths, basestring):
        paths = sorted(glob(paths))
    if not paths:
        raise IOError('no files to open')

    if lock is None:
        lock = _default_lock(paths[0], engine)
    datasets = [_open_dataset(p, engine=engine, chunks=chunks or {},
                              lock=lock, **kwargs) for p in paths]
    file_objs = [ds._file_obj for ds in datasets if ds is not None]

    if isinstance(concat_dim, pd.Index):
        name = concat_dim.name
        concat_dim = concat_dim.take(
            [ind for ind, ds in enumerate(datasets) if ds is not None])
        concat_dim.name = name

    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets if ds is not None]

    if concat_dim is _CONCAT_DIM_DEFAULT:
        combined = auto_combine(datasets, compat=compat)
    else:
        combined = auto_combine(datasets, concat_dim=concat_dim,
                                compat=compat)
    combined._file_obj = _MultiFileCloser(file_objs)
    combined.attrs = datasets[0].attrs

    return combined


def read_analysis_files(processed_data_dir, epoch_keys, **kwargs):
    '''Reads in analysis files and concatenate them.
    '''
    epoch_keys.name = 'recording_session'
    file_names = [get_analysis_file_path(processed_data_dir, *epoch_key)
                  for epoch_key in epoch_keys]
    return open_mfdataset(
        file_names, concat_dim=epoch_keys, **kwargs)
