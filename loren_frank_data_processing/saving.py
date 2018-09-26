from os.path import isfile, join


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
