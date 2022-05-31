import tempfile
import unittest
from os.path import join

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
from skimage.data._fetchers import camera, microaneurysms

from faim_dl.histogram.UIntHistogram import UIntHistogram


class TestUIntHistogram(unittest.TestCase):

    def test_bins(self):
        data = np.array([4, 5, 6])
        hist = UIntHistogram(data)
        assert hist.bins == len(hist.frequencies) + 1

    def test_update_same_length(self):
        data = np.array([4, 5, 6])
        hist = UIntHistogram(data)
        hist.update(data)

        assert_array_equal(hist.frequencies, np.array([2, 2, 2]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

    def test_update_lower_overlap(self):
        data = np.array([4, 5, 6])
        update_data = np.array([2, 3, 4])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 1, 2, 1, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 2)

        update_data = np.array([2, 3, 4, 5, 6])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 1, 2, 2, 2]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 2)

    def test_update_lower_concat(self):
        data = np.array([4, 5, 6])
        update_data = np.array([1, 2, 3])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 1, 1, 1, 1, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 1)

    def test_update_lower_gap(self):
        data = np.array([4, 5, 6])
        update_data = np.array([1, 2])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 1, 0, 1, 1, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 1)

    def test_update_upper_overlap(self):
        data = np.array([4, 5, 6])
        update_data = np.array([5, 6, 7])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 2, 2, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

        update_data = np.array([4, 5, 6, 7])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([2, 2, 2, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

    def test_update_upper_concat(self):
        data = np.array([4, 5, 6])
        update_data = np.array([7, 8])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 1, 1, 1, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

    def test_update_upper_gap(self):
        data = np.array([4, 5, 6])
        update_data = np.array([8, 9])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 1, 1, 0, 1, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

    def test_update_covered(self):
        data = np.array([4, 5, 6])
        update_data = np.array([5])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 2, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

        data = np.array([4, 5, 6, 7])
        update_data = np.array([5, 6])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 2, 2, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

        data = np.array([4, 5, 6, 7])
        update_data = np.array([4, 5, 6])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([2, 2, 2, 1]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

        data = np.array([4, 5, 6, 7])
        update_data = np.array([5, 6, 7])
        hist = UIntHistogram(data)
        hist.update(update_data)

        assert_array_equal(hist.frequencies, np.array([1, 2, 2, 2]))
        assert_equal(hist.bins, len(hist.frequencies) + 1)
        assert_equal(hist.offset, 4)

    def test_mean(self):
        data = np.array(camera())
        hist = UIntHistogram(data)
        assert_almost_equal(hist.mean(), np.mean(data), decimal=10)

        update_data = microaneurysms()
        hist.update(update_data)
        assert_almost_equal(hist.mean(), np.mean(np.concatenate([data.ravel(), update_data.ravel()])), decimal=10)

    def test_std(self):
        data = np.array(camera())
        hist = UIntHistogram(data)
        assert_almost_equal(hist.std(), np.std(data), decimal=10)

        update_data = microaneurysms()
        hist.update(update_data)
        assert_almost_equal(hist.std(), np.std(np.concatenate([data.ravel(), update_data.ravel()])), decimal=10)

    def test_quantile(self):
        data = np.array(camera())

        hist = UIntHistogram(data)
        assert_equal(hist.quantile(0.), np.quantile(data, 0.))
        assert_equal(hist.quantile(0.25), np.quantile(data, 0.25))
        assert_equal(hist.quantile(0.5), np.quantile(data, 0.5))
        assert_equal(hist.quantile(0.75), np.quantile(data, 0.75))
        assert_equal(hist.quantile(1.), np.quantile(data, 1.))

        update_data = microaneurysms()
        hist.update(update_data)
        assert_equal(hist.quantile(0.), np.quantile(np.concatenate([data.ravel(), update_data.ravel()]), 0.))
        assert_equal(hist.quantile(0.25), np.quantile(np.concatenate([data.ravel(), update_data.ravel()]), 0.25))
        assert_equal(hist.quantile(0.5), np.quantile(np.concatenate([data.ravel(), update_data.ravel()]), 0.5))
        assert_equal(hist.quantile(0.75), np.quantile(np.concatenate([data.ravel(), update_data.ravel()]), 0.75))
        assert_equal(hist.quantile(1.), np.quantile(np.concatenate([data.ravel(), update_data.ravel()]), 1.))

    def test_min(self):
        data = np.array(camera())
        hist = UIntHistogram(data)
        assert_almost_equal(hist.min(), np.min(data), decimal=10)

        update_data = microaneurysms()
        hist.update(update_data)
        assert_almost_equal(hist.min(), np.min(np.concatenate([data.ravel(), update_data.ravel()])), decimal=10)

    def test_max(self):
        data = np.array(camera())
        hist = UIntHistogram(data)
        assert_almost_equal(hist.max(), np.max(data), decimal=10)

        update_data = microaneurysms()
        hist.update(update_data)
        assert_almost_equal(hist.max(), np.max(np.concatenate([data.ravel(), update_data.ravel()])), decimal=10)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = np.array([4, 5, 6])
            hist = UIntHistogram(data)

            hist.save(join(temp_dir, "hist.npz"))

            hist_ = UIntHistogram.load(join(temp_dir, "hist.npz"))

            assert_equal(hist_.bins, hist.bins)
            assert_equal(hist_.offset, hist.offset)
            assert_array_equal(hist_.frequencies, hist.frequencies)

            assert hist != hist_


if __name__ == '__main__':
    unittest.main()
