import numpy as np
import pytest

from python.gestures.data_loader.tiny_data import data_paths, load_tiny_data

DATA_DIR = "/Users/netanelblumenfeld/Desktop/data/11G/"


@pytest.mark.parametrize(
    "data_dir, people, gestures, data_type, expected",
    [
        # Test case for npy data type
        (
            "/fake/dir",
            3,
            ["gesture1", "gesture2"],
            "npy",
            [
                "/fake/dir/data_npy/p1/gesture1_1s.npy",
                "/fake/dir/data_npy/p1/gesture2_1s.npy",
                "/fake/dir/data_npy/p2/gesture1_1s.npy",
                "/fake/dir/data_npy/p2/gesture2_1s.npy",
            ],
        ),
        # Test case for doppler data type
        (
            "/fake/dir",
            2,
            ["gesture1"],
            "doppler",
            [
                "/fake/dir/data_doppler/p1/gesture1_1s_wl32_doppl.npy",
            ],
        ),
        # Test case for zero people
        ("/fake/dir", 0, ["gesture1", "gesture2"], "npy", []),
        # Test case for no gestures
        ("/fake/dir", 2, [], "npy", []),
    ],
)
def test_data_paths(data_dir, people, gestures, data_type, expected):
    assert data_paths(data_dir, people, gestures, data_type) == expected


def test_load_tiny_data():
    X, y = load_tiny_data(DATA_DIR, 2, ["PinchIndex"], "npy", use_pool=True)
    assert X.shape[1:] == (5, 2, 32, 492)
    assert y.shape[1] == 5
    assert X.dtype == np.complex64
