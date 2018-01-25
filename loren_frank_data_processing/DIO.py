'''
The DIO cell gives arrival/departure times at the end of each arm of the maze
(as indicated by the IR motion sensors at the end of the wells) and the
start/stop times for the output trigger to the reward pump
'''

import numpy as np
import pandas as pd

from .core import get_data_structure
from .tetrodes import get_trial_time


def get_DIO(epoch_key, animals):
    ''''''
    animal, day, epoch = epoch_key
    pins = get_data_structure(
        animals[animal], day, 'DIO', 'DIO')[epoch - 1].squeeze()
    pins_df = []

    for pin in pins:
        try:
            try:
                time = pd.to_timedelta(pin['times'][0, 0].squeeze(), unit='s')
            except ValueError:
                time = pd.to_timedelta(pin['times'][0, 0].item(), unit='s')
            values = pin['values'][0, 0].squeeze()
            pin_id = pin['original_id'][0, 0].item()

            try:
                series = pd.Series(values, index=time, name=pin_id)
            except TypeError:
                series = pd.Series(name=pin_id)
            pins_df.append(series)
        except IndexError:
            continue
    return pd.concat(pins_df, axis=1).fillna(0).sort_index()


def get_DIO_indicator(epoch_key, animals, time_function=get_trial_time):
    time = time_function(epoch_key, animals)
    dio = get_DIO(epoch_key, animals)
    time_index = np.digitize(dio.index.total_seconds(),
                             time.total_seconds())
    time_index[time_index == time.size] = time.size - 1
    return (dio.groupby(time[time_index]).sum()
            .reindex(index=time, fill_value=0))
