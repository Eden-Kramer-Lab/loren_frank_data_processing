from .core import find_closest_ind, get_data_structure


def get_DIO_variable(animal, days, dio_var, epoch_type='', environment=''):
    '''Returns a list of lists given a DIO variable (pulsetimes, timesincelast,
    pulselength, and pulseind) with a length corresponding to the number of
    epochs (first level) and the number of active pins (second level)
    '''
    epoch_pins = get_data_structure(animal, days, 'DIO', 'DIO',
                                    epoch_type=epoch_type,
                                    environment=environment)
    return [
        [pin[0][dio_var][0][0] for pin in pins.T
         if pin[0].dtype.names is not None]
        for pins in epoch_pins
    ]


def get_pulse_position_ind(pulse_times, position_times):
    '''Returns the index of a pulse from the DIO data structure in terms of the
    position structure time.
    '''
    # position times from the pos files are already adjusted
    TIME_CONSTANT = 1E4
    return [find_closest_ind(
        position_times, pin_pulse_times[:, 0].flatten() / TIME_CONSTANT)
        for pin_pulse_times in pulse_times]
