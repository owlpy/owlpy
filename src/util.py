# -----------------------------------------------------------------------------
# OwlPy - AGPLv3
#
# This file is part of the OwlPy library. For licensing information see the
# accompanying file `LICENSE`.
#
# The OwlPy Developers, 21st century
# -----------------------------------------------------------------------------


import math
import numpy as np

try:
    from pyrocko.trace import Trace as PTrace
except ImportError:
    PTrace = None

try:
    from obspy import Trace as OTrace
except ImportError:
    OTrace = None


from .error import OwlPyError


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
            tr.stats.starttime.timestamp)
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
        :py:class:`~owlpy.error.OwlPyError` if traces have different time
        span, sample rate or data type, or if traces is an empty list.

    :returns:
        2D array as ``data[itrace, isample]``.
    :rtype:
        :py:class:`numpy.ndarray`
    '''

    if not traces:
        raise OwlPyError('Need at least one trace.')

    udata = [_unpack_trace(tr) for tr in traces]

    params = [
        (data.size, data.dtype, deltat, tmin)
        for (data, deltat, tmin)
        in udata]

    if not _all_same(params):

        raise OwlPyError(
            'Given traces are incompatible. Unable to join multiple '
            'components into a single 2D array. Sampling rate, start time, '
            'number of samples and data type must match.\n%s\n%s' % (
                '  %10s %-10s %12s %22s' % (
                    'samples', 'dtype', 'deltat', 'tmin'),
                '\n'.join(
                    '  %10i %-10s %12.5e %22.16e' % vec for vec in params)))

    return np.vstack([data for (data, _, _) in udata])


class ArangeError(Exception):
    pass


def arange2(start, stop, step, dtype=float, epsilon=1e-6, error='raise'):
    '''
    Return evenly spaced numbers over a specified interval.

    Like :py:func:`numpy.arange` but returning floating point numbers by
    default and with defined behaviour when stepsize is inconsistent with
    interval bounds. It is considered inconsistent if the difference between
    the closest multiple of ``step`` and ``stop`` is larger than ``epsilon *
    step``. Inconsistencies are handled according to the ``error`` parameter.
    If it is set to ``'raise'`` an exception of type :py:exc:`ArangeError` is
    raised. If it is set to ``'round'``, ``'floor'``, or ``'ceil'``, ``stop``
    is silently changed to the closest, the next smaller, or next larger
    multiple of ``step``, respectively.

    This function has been adapted from Pyrocko (pyrocko.util.arange2).
    '''

    assert error in ('raise', 'round', 'floor', 'ceil')

    start = dtype(start)
    stop = dtype(stop)
    step = dtype(step)

    rnd = {'floor': math.floor, 'ceil': math.ceil}.get(error, round)

    n = int(rnd((stop - start) / step)) + 1
    stop_check = start + (n-1) * step

    if error == 'raise' and abs(stop_check - stop) > step * epsilon:
        raise ArangeError(
            'inconsistent range specification: start=%g, stop=%g, step=%g'
            % (start, stop, step))

    x = np.arange(n, dtype=dtype)
    x *= step
    x += start
    return x


def moving_sum(x, n, mode='valid'):
    n = int(n)
    cx = np.cumsum(x, axis=-1)
    nn = x.shape[-1]

    def xzeros(n):
        return np.zeros(shape=x.shape[:-1] + (n,), dtype=cx.dtype)

    if mode == 'valid':
        if nn-n+1 <= 0:
            return xzeros(0)

        y = xzeros(nn-n+1)
        y[..., 0] = cx[..., n-1]
        y[..., 1:nn-n+1] = cx[..., n:nn]-cx[..., 0:nn-n]

    if mode == 'full':
        y = xzeros(nn+n-1)
        if n <= nn:
            y[..., 0:n] = cx[..., 0:n]
            y[..., n:nn] = cx[..., n:nn] - cx[..., 0:nn-n]
            y[..., nn:nn+n-1] = cx[..., -1]-cx[..., nn-n:nn-1]
        else:
            y[..., 0:nn] = cx[..., 0:nn]
            y[..., nn:n] = cx[..., nn-1]
            y[..., n:nn+n-1] = cx[..., nn-1] - cx[..., 0:nn-1]

    if mode == 'same':
        n1 = (n-1)//2
        y = xzeros(nn)
        if n <= nn:
            y[..., 0:n-n1] = cx[..., n1:n]
            y[..., n-n1:nn-n1] = cx[..., n:nn]-cx[..., 0:nn-n]
            y[..., nn-n1:nn] = cx[..., nn-1, np.newaxis] \
                - cx[..., nn-n:nn-n+n1]
        else:
            y[..., 0:max(0, nn-n1)] = cx[..., min(n1, nn):nn]
            y[..., max(nn-n1, 0):min(n-n1, nn)] = cx[..., nn-1]
            y[..., min(n-n1, nn):nn] = cx[..., nn-1] \
                - cx[..., 0:max(0, nn-(n-n1))]

    return y
