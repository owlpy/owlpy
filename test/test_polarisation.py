# -----------------------------------------------------------------------------
# OwlPy - AGPLv3
#
# This file is part of the OwlPy library. For licensing information see the
# accompanying file `LICENSE`.
#
# The OwlPy Developers, 21st century
# -----------------------------------------------------------------------------


import os
import math
import numpy as np

from pyrocko import trace as ptrace, util as putil
from pyrocko import moment_tensor as pmt
from pyrocko import gf
from pyrocko.obspy_compat import to_obspy_trace

from owlpy.polarisation import pca
from owlpy.polarisation import gridsearch
from owlpy.util import get_traces_data_as_array

d2r = np.pi/180.
km = 1000.


show_plot = int(os.environ.get('MPL_SHOW', 0))


def angle_sub(a, b, period=360.):
    return ((a-b)+0.5*period) % period - 0.5*period


def isclose_angle(a, b, period=360., **kwargs):
    return math.isclose(angle_sub(a, b, period=period), 0., **kwargs)


def arr(x):
    return np.array(x, dtype=float)


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


def make_synthetic_signal(
        mt=pmt.MomentTensor(strike=0., dip=90., rake=0),
        distance=6000*km,
        source_depth=10e3,
        azimuth=45.):

    engine = gf.LocalEngine(store_superdirs=['gf_stores'])

    mnn, mee, mdd, mne, mnd, med = mt.m6()

    source = gf.MTSource(
        mnn=mnn, mee=mee, mdd=mdd, mne=mne, mnd=mnd, med=med,
        north_shift=distance * np.cos((180.+azimuth)*d2r),
        east_shift=distance * np.sin((180.+azimuth)*d2r),
        depth=source_depth)

    store_id = 'test_qssp_fd_%.0f' % (distance/km)

    store = engine.get_store(store_id)

    assert store.config.distance_delta == store.config.source_depth_delta
    assert store.config.distance_delta == store.config.receiver_depth_delta
    delta = store.config.distance_delta

    comp_azi = {
        'N': 0.0,
        'E': 90.0,
        'D': 0.0}

    comp_dip = {
        'N': 0.0,
        'E': 0.0,
        'D': 90.0}

    traces = []
    for kn in [-1, 0, 1]:
        for ke in [-1, 0, 1]:
            for kd in [0, 1]:
                targets = [gf.Target(
                    codes=('', 'STA', '', comp),
                    quantity='displacement',
                    depth=kd * delta,
                    north_shift=kn * delta,
                    east_shift=ke * delta,
                    azimuth=comp_azi[comp],
                    dip=comp_dip[comp],
                    interpolation='multilinear',
                    store_id=store_id) for comp in 'NED']

                response = engine.process(source, targets)
                traces.extend(response.pyrocko_traces())

    fnyquist = 0.5 * store.config.sample_rate
    fmax = 0.8 * fnyquist
    tpad = 2.0 / fmax
    tmin = min(tr.tmin for tr in traces) - tpad
    tmax = max(tr.tmax for tr in traces) + tpad
    traces_prepared = []
    for tr in traces:
        tr.extend(tmin, tmax, fillmethod='repeat')
        tr_filtered = tr.transfer(
            tfade=tpad, freqlimits=(-1., -1., fmax, fnyquist))
        traces_prepared.append(tr_filtered)

    d = get_traces_data_as_array(traces)
    d = d.reshape((3, 3, 2, 3, d.shape[1]))

    kn, ke, kd = 0, 1, 2
    il, ic, ih = 0, 1, 2
    kc, kh = 0, 1
    de_dd = (d[ic, ic, kh, ke] - d[ic, ic, kc, ke]) / delta
    dd_de = (d[ic, ih, kc, kd] - d[ic, il, kc, kd]) / (2.*delta)

    dd_dn = (d[ih, ic, kc, kd] - d[il, ic, kc, kd]) / (2.*delta)
    dn_dd = (d[ic, ic, kh, kn] - d[ic, ic, kc, kn]) / delta

    dn_de = (d[ic, ih, kc, kn] - d[ic, il, kc, kn]) / (2.*delta)
    de_dn = (d[ih, ic, kc, ke] - d[il, ic, kc, ke]) / (2.*delta)

    d_rot = [
        0.5 * (de_dd - dd_de),
        0.5 * (dd_dn - dn_dd),
        0.5 * (dn_de - de_dn)]

    traces_rot = []
    for comp, data in zip('NED', d_rot):
        tr = traces[0].copy()
        tr.set_ydata(data)
        tr.set_codes(station='ROT', channel=comp)
        tr.differentiate(1)
        traces_rot.append(tr)

    traces_dis, traces_vel, traces_acc = [], [], []
    for comp, data in zip('NED', d[ic, ic, kc, :]):
        tr_dis = traces[0].copy()
        tr_dis.set_ydata(data)
        tr_dis.set_codes(station='DIS', channel=comp)
        tr_vel = tr_dis.differentiate(1, inplace=False)
        tr_vel.set_codes(station='VEL')
        tr_acc = tr_dis.differentiate(2, inplace=False)
        tr_acc.set_codes(station='ACC')
        traces_dis.append(tr_dis)
        traces_vel.append(tr_vel)
        traces_acc.append(tr_acc)

    return traces_dis, traces_vel, traces_acc, traces_rot


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


def test_gridsearch_azimuth_rot_acc():
    azimuth = 45.
    dis, vel, acc, rot = make_synthetic_signal(
        mt=pmt.MomentTensor.random_dc(), azimuth=azimuth)

    def ampmax(traces):
        return max(np.max(np.abs(tr.ydata)) for tr in traces)

    amax_acc = ampmax(acc)
    amax_rot = ampmax(rot)

    for tr in acc:
        tr.ydata += amax_acc * 0.02 * np.random.normal(size=tr.ydata.size)
        tr.lowpass(4, 0.2)

    for tr in rot:
        tr.ydata += amax_rot * 0.02 * np.random.normal(size=tr.ydata.size)
        tr.lowpass(4, 0.2)

    times, azimuths, correlations, max_azimuths, max_correlations = \
        gridsearch.gridsearch_azimuth_rot_acc(
            [rot[-1], acc[0], acc[1]], time_sum=40.)

    # ptrace.snuffle(acc + rot)

    if show_plot:
        from matplotlib import pyplot as plt

        def draw_trace(axes, tr, ipos, amax, **kwargs):
            t = tr.get_xdata()
            y = ipos + (0.5 / amax) * tr.get_ydata()
            axes.plot(t, y, **kwargs)

        fig = plt.figure()
        axes = fig.add_subplot(4, 1, 1)

        amax_rot = ampmax(rot)
        amax_acc = ampmax(acc)
        draw_trace(axes, rot[0], 5, amax_rot, color=(0.7, 0.2, 0.2))
        draw_trace(axes, rot[1], 4, amax_rot, color=(0.7, 0.2, 0.2))
        draw_trace(axes, rot[2], 3, amax_rot, color=(0.7, 0.2, 0.2))
        draw_trace(axes, acc[0], 2, amax_acc, color='black')
        draw_trace(axes, acc[1], 1, amax_acc, color='black')
        draw_trace(axes, acc[2], 0, amax_acc, color='black')

        axes = fig.add_subplot(4, 1, 2, sharex=axes)
        axes.pcolormesh(times, azimuths, correlations)

        axes = fig.add_subplot(4, 1, 3, sharex=axes)
        axes.plot(times, max_correlations, color='black')

        axes = fig.add_subplot(4, 1, 4, sharex=axes)
        axes.axhline(azimuth, color=(0.7, 0.2, 0.2))
        axes.scatter(times, max_azimuths, c=max_correlations, cmap='Greys')

        plt.show()
