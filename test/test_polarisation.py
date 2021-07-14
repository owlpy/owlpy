# -----------------------------------------------------------------------------
# Pyrots - AGPLv3
#
# This file is part of the Pyrots library. For licensing information see the
# accompanying file `LICENSE`.
#
# The Pyrots Developers, 21st century
# -----------------------------------------------------------------------------


import math
import numpy as np

from pyrocko import trace as ptrace, util as putil
from pyrocko.obspy_compat import to_obspy_trace

from pyrots.polarisation import pca

d2r = np.pi/180.


def angle_sub(a, b, period=360.):
    return ((a-b)+0.5*period) % period - 0.5*period


def isclose_angle(a, b, period=360., **kwargs):
    return math.isclose(angle_sub(a, b, period=period), 0., **kwargs)


def make_noisy_polarized_signal(
        tmin=0.0,
        deltat=0.01,
        nsamples=1000,
        azimuth=30.,
        incidence=90.,
        amp_noise=1.0,
        amp_signal=1.0):

    data_noise = amp_noise * np.random.normal(size=(3, nsamples))
    signal = amp_signal * np.random.normal(size=nsamples)
    data_signal = np.zeros((3, nsamples))
    data_signal[0, :] = np.sin(incidence*d2r) * np.sin(azimuth*d2r) * signal
    data_signal[1, :] = np.sin(incidence*d2r) * np.cos(azimuth*d2r) * signal
    data_signal[2, :] = np.cos(incidence*d2r) * signal
    data = data_noise + data_signal

    return [
        ptrace.Trace(
            '', 'STA', '', comp,
            tmin=tmin,
            deltat=deltat,
            ydata=ydata)
        for comp, ydata in zip('ENZ', data)]


def test_pca():
    incidence_in = 70.
    for azimuth_in in putil.arange2(-180., 180., 30.):
        trs = make_noisy_polarized_signal(
            amp_noise=0.1,
            azimuth=azimuth_in,
            incidence=incidence_in)

        # ptrace.snuffle(trs)

        for trs_ in trs, [to_obspy_trace(tr) for tr in trs]:
            cov, evals, evecs, azimuth_out1, incidence_out1 = pca.pca(trs[:2])
            azimuth_out2, incidence_out2 = pca.pca(trs)[-2:]
            assert 0. <= azimuth_out1 < 180.
            assert 0. <= azimuth_out2 < 180.
            assert 0. <= incidence_out1 <= 90.
            assert 0. <= incidence_out2 <= 90.
            assert isclose_angle(
                azimuth_in, azimuth_out1, period=180., abs_tol=5.0)
            assert isclose_angle(
                azimuth_in, azimuth_out2, period=180., abs_tol=5.0)
            assert incidence_out1 == 90.
            assert isclose_angle(incidence_out2, incidence_in, abs_tol=5.0)
