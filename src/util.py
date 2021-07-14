# -----------------------------------------------------------------------------
# Pyrots - AGPLv3
#
# This file is part of the Pyrots library. For licensing information see the
# accompanying file `LICENSE`.
#
# The Pyrots Developers, 21st century
# -----------------------------------------------------------------------------


import numpy as np

try:
    from pyrocko.trace import Trace as PTrace
except ImportError:
    PTrace = None

try:
    from obspy import Trace as OTrace
except ImportError:
    OTrace = None


from .error import PyrotsError


def _all_same(xs):
    return all(x == xs[0] for x in xs)


def _unpack_trace(tr):
    '''
    Get trace attributes in a framework agnostic way.
    '''

    if PTrace and isinstance(tr, PTrace):
        return (
            tr.ydata,
            tr.deltat,
            tr.tmin)

    elif OTrace and isinstance(tr, OTrace):
        return (
            tr.data,
            1.0/tr.stats.sampling_rate,
            tr.starttime.timestamp)
    else:
        raise TypeError(
            'Expected ObsPy or Pyrocko trace but got object of type "%s".'
            % str(type(tr)))


def get_traces_data_as_array(traces):
    '''
    Merge data samples from multiple traces into a 2D array.

    :param traces:
        Input waveforms.
    :type traces:
        list of :py:class:`obspy.Trace <obspy.core.trace.Trace>`
        or :py:class:`pyrocko.Trace <pyrocko.trace.Trace>` objects

    :raises:
        :py:class:`~pyrots.error.PyrotsError` if traces have different time
        span, sample rate or data type, or if traces is an empty list.

    :returns:
        2D array as ``data[itrace, isample]``.
    :rtype:
        :py:class:`numpy.ndarray`
    '''

    if not traces:
        raise PyrotsError('Need at least one trace.')

    udata = [_unpack_trace(tr) for tr in traces]

    if not _all_same([
            (data.size, data.dtype, deltat, tmin)
            for (data, deltat, tmin)
            in udata]):

        raise PyrotsError(
            'Given traces are incompatible. Unable to join multiple '
            'components into a single 2D array. Sampling rate, start time, '
            'number of samples and data type must match.')

    return np.vstack([data for (data, _, _) in udata])
