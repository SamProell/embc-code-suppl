
import numpy as np


def heartrate_from_indices(indices, f, max_std_seconds=float("inf"),
                           min_num_peaks=2, use_median=False):
    """Calculate heart rate from given peak indices

    Args:
        indices (`np.ndarray`): indices of detected peaks
        f (float): in Hz; sampling rate of BCG signal
        min_num_peaks (int): minimum number of peaks to consider valid
        max_std_seconds (float): in seconds; maximum standard deviation
            of peak distances
        use_median (bool): calculate heart rate with median instead of
            mean

    Returns:
        float: mean heartrate estimation in beat/min
    """

    if len(indices) < min_num_peaks:
        return -1

    diffs = np.diff(indices).astype(float) / f
    if np.std(diffs) > max_std_seconds:
        return -1

    if use_median:
        return 60. / np.median(diffs)

    return 60. / np.mean(diffs)


def get_heartrate_pipe(segmenter, max_std_seconds=float("inf"), min_num_peaks=2,
                       use_median=False, index=None):
    """build function that estimates heart rate from detected peaks in
    input signal

    If stddev of peak distances exceeds `max_std_seconds` or less than
    `min_.num_peaks` peaks are found, input signal is marked as invalid
    by returning -1.
    If the `segmenter` returns tuples of wave indices (e.g. IJK instead
    of just J) the wave used for calculations has to be specified with
    `index`.

    Args:
        segmenter (function): BCG segmentation algorithm
        max_std_seconds (float): maximum stddev of peak distances
        min_num_peaks (int): minimum number of detected peaks
        use_median (bool): calculate heart rate from median of peak
            distances instead of mean
        index (int): index of wave used for calculations

    Returns:
        `function`: full heart rate estimation algorithm
    """

    def pipe(x, f, **args):
        indices = segmenter(x, f, **args)
        if index is not None:
            indices = indices[:, index]
        hr = heartrate_from_indices(indices, f,
                                    max_std_seconds=max_std_seconds,
                                    min_num_peaks=min_num_peaks,
                                    use_median=use_median)
        return hr

    return pipe


def get_heartrate_score_pipe(segmenter, use_median=False, index=None):
    """build function that estimates heart rate from detected peaks in
    input signal and return both heart rate and stddev of peak distances

    If the `segmenter` returns tuples of wave indices (e.g. IJK instead
    of just J) the wave used for calculations has to be specified with
    `index`.

    Args:
        segmenter (function): BCG segmentation algorithm
        use_median (bool): calculate heart rate from median of peak
            distances instead of mean
        index (int): index of wave used for calculations

    Returns:
        `function`: full heart rate estimation algorithm that returns
        both estimated heart rate and stddev of peak distances for given
        signal
    """

    def pipe(x, f, **args):
        indices = segmenter(x, f, **args)
        if index is not None:
            indices = indices[:, index]
        n = len(indices)
        if n < 2:
            return -1, -1
        diffs_std = np.std(np.diff(indices) / f)
        hr = heartrate_from_indices(indices, f, max_std_seconds=float("inf"),
                                    min_num_peaks=2, use_median=use_median)

        return hr, diffs_std

    return pipe
