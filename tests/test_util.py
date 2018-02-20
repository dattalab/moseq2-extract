import pytest
import os
import numpy as np
import numpy.testing as npt
import json
from moseq2.util import gen_batch_sequence, load_metadata, load_timestamps


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


def test_gen_batch_sequence():

    tmp_list = [range(0, 10),
                range(5, 15),
                range(10, 20),
                range(15, 25)]

    gen_list = list(gen_batch_sequence(25, 10, 5))

    assert(gen_list == tmp_list)


def test_load_timestamps(temp_dir):

    tmp_timestamps = np.arange(0, 5, .05)
    txt_file = os.path.join(temp_dir, 'test_timestamps.txt')

    with open(txt_file, 'w') as f:
        for timestamp in tmp_timestamps:
            print('{}'.format(str(timestamp)), file=f)

    loaded_timestamps = load_timestamps(txt_file)
    npt.assert_almost_equal(loaded_timestamps, tmp_timestamps, 10)


def test_load_metadata(temp_dir):

    tmp_dict = {
        'test': 'test2'
    }

    json_file = os.path.join(temp_dir, 'test_json.json')
    with open(json_file, 'w') as f:
        json.dump(tmp_dict, f)

    loaded_dict = load_metadata(json_file)

    assert(loaded_dict == tmp_dict)
