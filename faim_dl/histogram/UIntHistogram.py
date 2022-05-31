import numpy as np
from matplotlib import pyplot as plt


class UIntHistogram:
    def __init__(self, data=None):
        """
        An unsigned integer histogram class that can be updated with new data.

        :param data: array[uint]
            Optional data of which the initial histogram is built.
        """
        if data is not None:
            assert data.min() >= 0, 'Negative data is not supported.'
            self.offset, self.bins, self.frequencies = self._get_hist(data)
        else:
            self.offset, self.bins, self.frequencies = None, None, None

    @staticmethod
    def _add(list_a, list_b):
        """
        Add two lists element-wise.
        :param list_a: list(numeric)
        :param list_b: list(numeric)
        :return: list(numeric)
        """
        return list(map(lambda e: e[0] + e[1], zip(list_a, list_b)))

    @staticmethod
    def _get_hist(data):
        """
        Compute histogram with integer bins.
        :param data: array[numeric]
        :return: offset, bins, frequencies
        """
        offset = int(data.min())
        bins = int(data.max()) + 2 - offset
        freq = list(np.histogram(data, np.arange(offset, offset + bins))[0])
        return offset, bins, freq

    def update(self, data):
        """
        Update histogram by adding more data.

        :param data: array(uint)
            Data to be added to the histogram.
        """
        assert data.min() >= 0, 'Negative data is not supported.'

        if self.frequencies is None:
            self.offset, self.bins, self.frequencies = self._get_hist(data)
        else:
            offset_data, bins, freq = self._get_hist(data)

            lower_shift = offset_data - self.offset
            upper_shift = self.offset + self.bins - (offset_data + bins)

            if lower_shift == 0 and upper_shift == 0:
                # Old and new frequencies cover the same range:
                # [old frequencies]
                # [new frequencies]
                self.frequencies = self._add(self.frequencies, freq)
            elif lower_shift < 0 and (offset_data + bins - 1 >= self.offset) and (
                    offset_data + bins - 1 <= self.offset + self.bins - 1):
                # New frequencies have additional lower ones.
                #     [old frequencies]
                # [new frequencies]
                self.frequencies[:(offset_data + bins - self.offset - 1)] = self._add(
                    self.frequencies[:(offset_data + bins - self.offset - 1)],
                    freq[(self.offset - offset_data):])
                self.frequencies = freq[:(self.offset - offset_data)] + self.frequencies
                self.offset = offset_data
                self.bins = len(self.frequencies) + 1
            elif lower_shift < 0 and (offset_data + bins - 1 < self.offset):
                # New frequencies only have additional lower ones.
                #                       [old frequencies]
                # [new frequencies]
                self.frequencies = freq + [0, ] * (self.offset - (offset_data + bins) + 1) + self.frequencies
                self.offset = offset_data
                self.bins = len(self.frequencies) + 1
            elif offset_data >= self.offset and offset_data <= (
                    self.offset + self.bins - 2) and offset_data + bins - 2 > (self.offset + self.bins - 2):
                # New frequencies have additional upper ones.
                #     [old frequencies]
                #            [new frequencies]
                self.frequencies[-(self.offset + self.bins - 1 - offset_data):] = self._add(
                    self.frequencies[-(self.offset + self.bins - 1 - offset_data):],
                    freq[:(self.offset + self.bins - offset_data)])
                self.frequencies = self.frequencies + freq[(self.offset + self.bins - offset_data - 1):]
                self.bins = len(self.frequencies) + 1
            elif (offset_data) > (self.offset + self.bins - 2):
                # New frequencies have only additional upper ones.
                #     [old frequencies]
                #                           [new frequencies]
                self.frequencies = self.frequencies + [0, ] * (offset_data - (self.offset + self.bins - 1)) + freq
                self.bins = len(self.frequencies) + 1
            elif lower_shift >= 0 and upper_shift >= 0:
                # New frequencies are completely covered.
                # [          old frequencies          ]
                #           [new frequencies]
                if upper_shift == 0:
                    self.frequencies[lower_shift:] = self._add(self.frequencies[lower_shift:], freq)
                else:
                    self.frequencies[lower_shift:-upper_shift] = self._add(self.frequencies[lower_shift:-upper_shift], freq)
            else:
                # Old frequencies are completely covered.
                #     [old frequencies]
                # [    new frequencies    ]
                self.frequencies = self._add(self.frequencies,
                                             freq[(self.offset - offset_data):(
                                                     (offset_data + bins - 1) - (self.offset + bins - 1))])
                self.frequencies = freq[:(self.offset - offset_data)] + self.frequencies + freq[((offset_data + bins) - (
                        self.offset + bins)):]
                self.offset = offset_data
                self.bins = len(self.frequencies) + 1

    def plot(self):
        """
        Plot histogram.
        """
        plt.bar(np.arange(self.offset, self.offset + self.bins - 1), self.frequencies, width=1)
        plt.show()

    def mean(self):
        """
        Get histogram mean.
        :return: float
        """
        return np.sum(np.arange(self.offset, self.offset + self.bins - 1) * self.frequencies) / np.sum(
            self.frequencies)

    def std(self):
        """
        Get histogram standard deviation.
        :return: float
        """
        return np.sqrt(np.sum(
            (np.arange(self.offset, self.offset + self.bins - 1) - self.mean()) ** 2 * self.frequencies) / np.sum(
            self.frequencies))

    def quantile(self, q):
        """
        Get quantile `q`.
        :param q: uint
        :return: quantile
        """
        assert q >= 0 and q <= 1
        return self.offset + np.argmax(np.cumsum(self.frequencies) / np.sum(self.frequencies) >= q)

    def min(self):
        """
        Get minimum.
        :return: uint
        """
        return self.offset

    def max(self):
        """
        Get maximum.
        :return: uint
        """
        return self.offset + self.bins - 2

    def save(self, path):
        np.savez(path, frequencies=self.frequencies, offset=self.offset, bins=self.bins)

    @staticmethod
    def load(path):
        storage = np.load(path)
        hist = UIntHistogram()
        hist.frequencies = storage["frequencies"]
        hist.offset = storage["offset"]
        hist.bins = storage["bins"]
        return hist
