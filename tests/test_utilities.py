import pytest
import numpy as np
from loren_frank_data_processing.utilities import find_closest_ind


@pytest.mark.parametrize(
    "search_array, target, expected_index",
    [
        (np.arange(50, 150), 66, 16),
        (np.arange(50, 150), 45, 0),
        (np.arange(50, 150), 200, 99),
        (np.arange(50, 150), 66.4, 16),
        (np.arange(50, 150), 66.7, 17),
        (np.arange(50, 150), [55, 65, 137], [5, 15, 87]),
    ],
)
def test_find_closest_ind(search_array, target, expected_index):
    assert np.all(find_closest_ind(search_array, target) == expected_index)
