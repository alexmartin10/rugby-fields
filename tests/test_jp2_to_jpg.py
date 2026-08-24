import numpy as np

from inference.jp2_to_jpg import compute_start_indices


row_starts_normal, col_starts_normal = compute_start_indices(25000, 25000, 2048, 0.2, False) #normal case in our project
row_starts_rectangle, col_starts_rectangle = compute_start_indices(4300, 2048, 2048, 0.2, False) #case where a hole could appear
row_starts_single, col_starts_single = compute_start_indices(2048, 2048, 2048, 0.2, False)


def test_normal():
    #indices start at zero
    assert row_starts_normal[0] == 0
    assert col_starts_normal[0] == 0

    #last start is at dimension - window_size
    assert row_starts_normal[-1] == 25000 - 2048
    assert col_starts_normal[-1] == 25000 - 2048

    #no hole between two windows
    assert np.all(np.diff(row_starts_normal) <= 2048)
    assert np.all(np.diff(col_starts_normal) <= 2048)

    #actual overlap is close to the target
    assert 0.18 <= 1 - row_starts_normal[1] / 2048 <= 0.22

def test_rectangle():
    assert row_starts_rectangle[0] == 0
    assert col_starts_rectangle[0] == 0

    assert len(row_starts_rectangle) == 3
    assert col_starts_rectangle.tolist() == [0]

    assert row_starts_rectangle[-1] == 4300 - 2048

    assert np.all(np.diff(row_starts_rectangle) <= 2048)
    assert np.all(np.diff(col_starts_rectangle) <= 2048)

def test_single():
    assert row_starts_single.tolist() == [0]
    assert col_starts_single.tolist() == [0]