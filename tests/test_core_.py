from collections import namedtuple
from unittest.mock import patch

import numpy as np

import pytest
from loren_frank_data_processing.core import get_data_filename, get_epochs

# Create a fake 'tasks' data set to test
mock_data_struct = np.zeros(
    5, dtype={"names": ["type", "environment"], "formats": ["O", "O"]}
)
mock_data_struct[0] = ("typeTest1", "environTest1")
mock_data_struct[1] = ("typeTest1", "environTest2")
mock_data_struct[2] = ("typeTest2", "environTest2")
mock_data_struct[3] = ("typeTest1", "environTest2")
mock_data_struct[4] = ("typeTest1", "environTest1")

MOCK_CELL_ARRAY = {"task": np.array([[[], [mock_data_struct], [mock_data_struct]]])}


@pytest.mark.parametrize(
    "day, expected_name",
    [
        (2, "/Raw-Data/test_dir/Testdummy02.mat"),
        (11, "/Raw-Data/test_dir/Testdummy11.mat"),
    ],
)
def test_data_file_name_returns_correct_file(day, expected_name):
    Animal = namedtuple("Animal", {"directory", "short_name"})
    animal = Animal(directory="/Raw-Data/test_dir", short_name="Test")
    file_type = "dummy"

    file_name = get_data_filename(animal, day, file_type)
    assert expected_name in file_name


@patch("loren_frank_data_processing.core.loadmat", return_value=MOCK_CELL_ARRAY)
def test_get_epochs(mock_loadmat):
    Animal = namedtuple("Animal", {"directory", "short_name"})
    animal = Animal(directory="/Raw-Data/test_dir", short_name="Test")
    day = 2
    expected_length = 5

    assert len(get_epochs(animal, day)) == expected_length
