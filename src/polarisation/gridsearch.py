# -----------------------------------------------------------------------------
# OwlPy - AGPLv3
#
# This file is part of the OwlPy library. For licensing information see the
# accompanying file `LICENSE`.
#
# The OwlPy Developers, 21st century
# -----------------------------------------------------------------------------

import numpy as np

from owlpy.util import get_traces_data_as_array, arange2, moving_sum

d2r = np.pi / 180.


def gridsearch_azimuth_rot_acc(traces, time_sum, azimuth_delta=5.):

    '''
    Get direction of SH/Love waves from rotational and acceleration waveforms.

    TODO: add method description

    :param traces:
        Waveforms of the signals to be analysed. Components are expected in the
        order and polarity ``[rotation_rate_down, accelaration_north,
        acceleration_east]``. The traces must be of same length, sampling rate
        and data type.
    :type traces:
        list of :py:class:`obspy.Trace <obspy.core.trace.Trace>`
        or :py:class:`pyrocko.Trace <pyrocko.trace.Trace>` objects

    :param time_sum:
        Length of gliding window for correlation determination [s].
    :type time_sum:
        float

    :param azimuth_delta:
        Azimuth grid step size [deg].
    :type azimuth_delta:
        float

    :raises:
        :py:exc:`~owlpy.error.OwlPyError` if the input traces are
        incompatible.
    '''

    trace_rot_z, trace_acc_n, trace_acc_e = traces

    data = get_traces_data_as_array([trace_rot_z, trace_acc_n, trace_acc_e])
    deltat = trace_rot_z.deltat

    irotz, iaccn, iacce = 0, 1, 2
    nsamples = data.shape[1]
    azimuths = arange2(0., 360. - azimuth_delta, azimuth_delta)
    grid = np.zeros((azimuths.size, 2, nsamples))
    for iazimuth, azimuth in enumerate(azimuths):
        data_rot = data[irotz, :]
        data_acc_t = -data[iaccn, :] * np.sin(azimuth*d2r) \
            + data[iacce, :] * np.cos(azimuth*d2r)
        grid[iazimuth, 0, :] = data_rot * data_acc_t
        grid[iazimuth, 1, :] = data_acc_t**2

    nsum = int(np.round(time_sum / deltat))

    sgrid = moving_sum(grid, nsum, mode='same')
    srotz = moving_sum(data[irotz, :]**2, nsum, mode='same')
    grid_correlations = sgrid[:, 0, :] \
        / (np.sqrt(sgrid[:, 1, :]) * np.sqrt(srotz)[np.newaxis, :])

    max_indices = np.argmax(grid_correlations, axis=0)
    max_correlations = grid_correlations[max_indices, np.arange(nsamples)]
    max_azimuths = azimuths[max_indices]

    times = trace_rot_z.tmin + np.arange(nsamples) * deltat
    return times, azimuths, grid_correlations, max_azimuths, max_correlations
