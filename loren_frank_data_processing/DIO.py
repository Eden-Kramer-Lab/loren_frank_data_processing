import numpy as np
import pandas as pd

from .core import get_data_structure
from .tetrodes import get_trial_time


def get_DIO(epoch_key, animals):
    ''''''
    animal, day, epoch = epoch_key
    pins = get_data_structure(animals[animal], day, 'DIO', 'DIO')[
        epoch - 1].squeeze()
    pins_df = []

    for pin in pins:
        time = pin['times'][0, 0].squeeze()
        if time.size > 1:
            time = pd.to_timedelta(time, unit='s')

            values = pin['values'][0, 0].squeeze()
            pin_id = pin['original_id'][0, 0].item()

            pins_df.append(
                pd.Series(values, index=time, name=pin_id))
    return pd.concat(pins_df, axis=1).fillna(0)


def get_DIO_indicator(epoch_key, animals, time_function=get_trial_time):
    time = time_function(epoch_key, animals)
    dio = get_DIO(epoch_key, animals)
    time_index = np.digitize(dio.index.total_seconds(),
                             time.total_seconds())
    return (dio.groupby(time[time_index]).sum()
            .reindex(index=time, fill_value=0))
