# -----------------------------------------------------------------------------
# Pyrots - AGPLv3
#
# This file is part of the Pyrots library. For licensing information see the
# accompanying file `LICENSE`.
#
# The Pyrots Developers, 21st century
# -----------------------------------------------------------------------------


import numpy as np

from pyrots.error import PyrotsError
from pyrots.util import get_traces_data_as_array


r2d = 180. / np.pi


class PCAError(PyrotsError):
    '''
    Raised when PCA failed.
    '''
    pass


def pca(traces):
    '''
    Perform principal component analysis (PCA) of 2- or multi-component signal.

    :param traces:
        Waveforms of the signals to be analysed. Components are expected in the
        order and polarity ``[east, north]`` or ``[east, north, up]``. The
        traces must be of same length, sampling rate and data type.
    :type traces:
        list of :py:class:`obspy.Trace <obspy.core.trace.Trace>`
        or :py:class:`pyrocko.Trace <pyrocko.trace.Trace>` objects

    :raises:
        :py:exc:`~pyrots.error.PyrotsError` if the input traces are
        incompatible, :py:exc:`PCAError` if the traces are too short.

    :returns:
        ``(cov, evals, evecs, azimuth, incidence)`` where ``cov`` is the
        covariance matrix of the data, ``evals`` are the eigenvalues, ``evecs``
        are the eigenvectors, ``azimuth`` is the horizontal direction of the
        principal component, measured clockwise from north and ``incidence`` is
        incidence angle of the principal component, measured from vertical. The
        azimuth is wrapped to the range ``[0, 180)`` because of its +-180 deg
        ambiguity, the incidence angle returned in the range ``[0, 90]``. An
        incidence angle of 90 deg is returned, if no vertical component is
        available. Both angles are returned in [deg].
    :rtype:
        5-:py:class:`tuple`: three :py:class:`numpy.ndarray` and two
        :py:class:`float`.

    PCA is useful to find the polarisation of a signal contained in a 2- or
    3-component seismic recording. This function estimates the covariance of
    the signal, the eigen-system of the covariance and the angles of the first
    principal component. Note that the polarity of the polarisation direction
    cannot be determined from the PCA alone.
    '''

    data = get_traces_data_as_array(traces)

    cov = np.cov(data)

    evals, evecs = np.linalg.eigh(cov)
    # evals are returned in ascending order

    # first principal component,
    pc = evecs[:, -1]

    eh = np.sqrt(pc[1]**2 + pc[0]**2)
    if len(traces) > 2:
        incidence = r2d * np.arctan2(eh, abs(pc[2]))
    else:
        incidence = 90.

    azimuth = r2d * np.arctan2(pc[1], pc[0])
    azimuth = ((90. - azimuth) + 180) % 360. - 180.
    azimuth %= 180.

    return cov, evals, evecs, float(azimuth), float(incidence)
