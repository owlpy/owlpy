#!/usr/bin/env python

import sys
import numpy as np
from obspy import read_inventory
from obspy.core import UTCDateTime, read, Trace
from obspy.signal.trigger import classic_sta_lta, plot_trigger


def get_data(
        stream1, stream2, utctime, duration,
        seis_channel, rot_channel,
        inventory, ch_r, ch_s):

    '''
    Read in data from two files and do basic pre-processing.

    1. sort the channels
    2. remove the response
    3. rotate to zne-system if required
    4. cut out reqired time span
    5. select source and reciever channles

    :param stream1:
        full path to data recorded on 'seis_channel'
    :type stream1:
        string
    :param stream2:
        full path to data recorded on 'rot_channel'
    :type stream2:
        string
    :param utctime:
        start time of record to be analysed,
    :type utctime:
        string
        format: YYYY-MM-DDThh:mm:ss
    :param duration:
        length of time span to by analysed in sec
    :type duration:
        float
    :param seis_channel:
        channel(s) containing seismometer recordings
    :type seis_channel:
        string
    :param rot_channel:
        channel(s) containing rotation rate recordings
    :type rot_channel:
        string
    :param inventory:
        path to *.xml file containing response information
    :type inventory:
        string
    :param ch_r:
        reciever channel (data to be corrected)
    :type ch_r:
        string
    :param ch_s:
        source channel (data to correct for)
    :type ch_s:
        string

    :returns:
        r, s
        r: resciever channel
        s: source channel
    :rtype:
        obspy.Stream, obspy.Stream
    '''
    # define some parameters
    # take some seconds before and after the series of steps
    p = 0.1
    dt = p * duration

    # read the inventory for meta data
    inv = read_inventory(inventory)

    # get the strat time right
    t = UTCDateTime(utctime)

    # define the seismometer and rotation sensor input channels
    chan1 = seis_channel  # TODO: not used and probably not needed?  # noqa
    chan2 = rot_channel   # noqa

    # -------------------------------------------------------------------------
    # process the classic seismometer records
    # 1. read in the records and sort the channels, detrend and taper
    sz1 = read(stream1, starttime=t-dt, endtime=t+duration+dt)
    sz1.sort()
    sz1.reverse()
    sz1.detrend("linear")
    sz1.detrend("demean")
    sz1.taper(0.1)

    # 2. remove response and out put velocity in m/s
    sz1.attach_response(inv)
    sz1.remove_response(water_level=60, output="VEL")

    # 3. rotate the components according to the orientation as documented in
    # the inventory
    sz1.rotate(method="->ZNE", inventory=inv, components=["ZNE"])

# asign samplingrate and number of samples for seismometer channels
    df1 = sz1[0].stats.sampling_rate
    npts1 = sz1[0].stats.npts

    # -------------------------------------------------------------------------
    # process the rotation rate records
    # 1. read in the records and sort the channels, detrend and taper
    sz2 = read(stream2, starttime=t-dt, endtime=t+duration+dt)
    sz2.sort()
    sz2.detrend("demean")
    sz2.taper(0.1)

    # 2. remove response (scale by sensitivity) to out put rotation rate in
    # rad/s
    sz2.attach_response(inv)
    sz2.remove_sensitivity()
    # 3. rotate the components according to the orientation as documented in
    # the inventory
    sz2.rotate(method="->ZNE", inventory=inv, components=["321"])

    # asign samplingrate and number of samples for seismometer channels
    df2 = sz2[0].stats.sampling_rate
    npts2 = sz2[0].stats.npts

    # -------------------------------------------------------------------------
    # trim to the original time window and taper again
    sz1.trim(t, t+duration)
    sz2.trim(t, t+duration)
    sz1.taper(0.1)
    sz2.taper(0.1)

    # -------------------------------------------------------------------------
    # do sanity checks
    # 1. check for sampling rate
    # 2. check for number of samples
    if df1 != df2:
        print("Sampling rates not the same, exit!!")
        sys.exit(1)

    if npts1 != npts2:
        print("Number of data points not the same, exit!!")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # return the reciever and the source channel as defined in the arguments
    r = sz1.select(channel=ch_r)
    s = sz2.select(channel=ch_s)

    return r, s


def trigger(tr1, a, b, d0, d1, c_on, c_off, start, stop, plot_flagg=False):
    '''
    This method searches for time spans when the steps are performed.
    STA/LTA- trigger is used to calculate the characteristic function.
    A constant offset can be applied bacause the steps are uniform.

    :param tr1:
        rotation rate recording containing steps
    :type tr1:
        obspy.Trace
    :param a:
        number of samples for short term average
    :type a:
        int
    :param b:
        number of samples for long term average
    :type b:
        int
    :param d0:
        threshold for trigger-on
    :type d0:
        float
    :param d1:
        threshold for trigger-off
    :type d1:
        float
    :param c_on:
        constant correction for trigger at the start of each step in sec
    :type c_on:
        float
    :param c_off:
        constant correction for trigger at the end of each step in sec
    :type c_off:
        float
    :param start:
        offset in sec to start searching for steps
    :type start:
        float
    :param stop:
        offset in sec to stop searching for steps
    :type stop:
        float

    :returns:
        on, off
        on: start time of each step
        off: end time of each step
    :rtype:
        list, list
    '''
    # define some parameters
    data1 = tr1.data
    # df1 = tr1.stats.sampling_rate

    # get the characteristic function
    cft1 = classic_sta_lta(data1, int(a), int(b))

    # you can plot it if you want
    if plot_flagg:
        plot_trigger(tr1, cft1, d0, d1)

    # find the on/off time stamps of each step
    _on = np.where(cft1 < d0)[0]
    _off = np.where(cft1 > d1)[0]

    on = []
    on0 = 0
    for i in range(len(_on)-1):
        if _on[i+1] - _on[i] > 1:
            trigg = _on[i]*tr1.stats.delta
            if trigg >= start and trigg <= stop:
                if np.abs((trigg + c_on)-on0) > 1.0:
                    on.append(trigg + c_on)
                    on0 = trigg + c_on
    off = []
    off0 = 0
    for i in range(len(_off)-1):
        if _off[i+1] - _off[i] > 1:
            trigg = _off[i]*tr1.stats.delta
            if trigg >= start and trigg <= stop:
                if np.abs((trigg + c_off)-off0) > 1.0:
                    off.append(trigg + c_off)
                    off0 = trigg + c_off

    return on, off


def find_nearest(t, data, on, off):
    '''
    This method finds the nearest sample in 'data' to 'on' and 'off'

    :param t:
        array containing timestamps of samples in data
    :type t:
        numpy.ndarray
    :param data:
        data array where the nearest samples should be found
    :type data:
        numpy.ndarray
    :param on:
        time stamp found with method 'trigger()'
    :type on:
        float
    :param off:
        time stamp found with method 'trigger()'
    :type off:
        float

    :returns:
        idx_on, idx_off, data_on, data_off
        idx_on: index of first sample in step
        idx_off: index of last sample in step
        data_on: corresponding data point
        data_off: corresponding data point
    :rtype:
        int, int, float, float
    '''
    idx_on = (np.abs(t-on)).argmin()
    idx_off = (np.abs(t-off)).argmin()
    return idx_on, idx_off, data[idx_on], data[idx_off]


def calc_residual_disp(tr1, on, off, r, theo=False):
    '''
    This method calculates the residual displacement (lateral displacement
    introduced by the tilt motion) which is left over after tilt correction.

    :param tr1:
        trace containing tilt corrected velocity recording
    :type tr1:
        obspy.Trace
    :param on:
        list of time stamps found with method 'trigger()'
    :type on:
        list
    :param off:
        list of time stamps found with method 'trigger()'
    :type off:
        list
    :param r:
        array containing therotetical residual displacement. This is only used
        to shift the traces
    :type r:
        numpy.ndarray
        to make a nicer plot.
    :param theo:
        set True if theoretical displacement is calculated
    :type theo:
        bolean

    :returns:
        time, disp, mean, sigma
        time: list containing time stamps of each step
        disp: list containing residual displacement for each step
        mean: geometric mean value of 'disp'
        sigma: standard deviation of 'disp'
    :rtype:
        list, list, float, float
    '''

    disp_tr = []
    disp = []
    time = []

    t = np.arange(len(tr1[0].data))/(tr1[0].stats.sampling_rate)

    for i in range(len(on)):
        idx_on, idx_off, d_0, d_1 = find_nearest(t, tr1[0].data, on[i], off[i])

        data = tr1[0].data[idx_on:idx_off]
        stats = tr1[0].stats
        stats.starttime = tr1[0].stats.starttime+idx_on*tr1[0].stats.delta
        tr = Trace(data=data, header=stats)

        # suppose that velocity is zero at the beginning and at the end of a
        # step
        if not theo:
            tr.detrend('linear')
        y0 = tr.data[0]
        tr.data = tr.data - y0

        # integrate to displacement
        tr.integrate()

        # shift the whole trace to make it comparable to theoretical
        # displacement
        y0 = tr.data[0]
        diff = (y0 - r[idx_on])
        tr.data = tr.data - diff

        disp.append(tr.data)
        time.append(t[idx_on:idx_off])

        disp_tr.append(np.abs(max(tr.data)-min(tr.data)))

    mean_tr = np.mean(disp_tr)
    sigma_tr = np.std(disp_tr)

    return time, disp, mean_tr, sigma_tr


def get_angle(st, on, off):
    '''
    This method calculates the absolute angle for each step

    :param st:
        stream containing integrated rotation rate data (angle)
    :type st:
        obspy.Stream
    :param on:
        list of time stamps found with method 'trigger()'
    :type on:
        list
    :param off:
        list of time stamps found with method 'trigger()'
    :type off:
        list

    :returns:
        array containing absolute angle for each step
    :rtype:
        numpy.ndarray
    '''
    t = np.arange(len(st[0].data))/(st[0].stats.sampling_rate)
    alpha = []
    for i in range(len(on)):
        idx_on, idx_off, d_0, d_1 = find_nearest(t, st[0].data, on[i], off[i])
        alpha.append(np.abs(d_0 - d_1))
    return np.asarray(alpha)


def theo_resid_disp(alpha0, l, h, dh, rr):
    '''
    This method calculates the theoretical residual displacement
    induced by a tilt movement of the angle alpha0

    :param alpha0:
        integrated rotation rate recording (angle)
    :type alpha0:
        numpy.ndarray
    :param l:
        horizontal distance between axis of rotation and center of seismometer
        [m]
    :type l:
        float
    :param h:
        vertical distance between bottom of seismometer and seismometer mass
        [m]
    :type h:
        float
    :param dh:
        vertical distance between bottom of seismometer and axis of rotation
        [m]
    :type dh:
        float

    :returns:
        array containing theoretical residual displacement
    :rtype:
        numpy.ndarray
    '''
    x = l * (1. - np.cos(alpha0))
    y = (dh + h) * np.cos((np.pi/2) - alpha0)
    r = -1*(x + y)
    c = np.sqrt(l**2 + (dh+h)**2) * rr**2
    return r, c


def calc_height_of_mass(disp, l, dh, alpha):
    '''
    This method calculates the vertical distance between the bottom
    of the seismometer and the seismometer mass from the residual displacement.

    :param disp:
        list containing residual displacements for each step from
        'calc_residual_disp()'
    :type disp:
        list
    :param l:
        horizontal distance between axis of rotation and center of seismometer
        [m]
    :type l:
        float
    :param dh:
        vertical distance between bottom of seismometer and axis of rotation
        [m]
    :type dh:
        float
    :param alpha:
        rotation angles for each step from 'get_angle()'
    :type alpha:
        numpy.ndarray

    :returns:
        mean and standard deviation of vertical distance between the bottom
               of the seismometer and the seismometer mass
    :rtype:
        float, float
    '''
    alpha0 = alpha
    X = l * (1. - np.cos(alpha0))
    A = disp - X
    B = A / np.cos(alpha0)
    h = np.tan((np.pi/2.) - alpha0) * B - dh

    return np.mean(h), np.std(h)


def p2r(radii, angles):
    '''
    This method converts "raduis, angle" representation of a complex number to
    "r + ij" representation

    :param radii:
        radius of "radius, angle" representation
    :type radii:
        numpy float or numy array
    :param angles:
        angle of "radius, angle" representation
    :type angles:
        numpy float or numy array

    :returns:
        "r + ij" representation
    :rtype:
        numpy complex or numpy.ndarray numpy complex
    '''
    return radii*np.exp(1j*angles)


def r2p(x):
    '''
    This method converts "r + ij" representation of a complex number to
    "raduis, angle" representation.

    :param x:
        "r + ij" representation
    :type x:
        numpy complex or numpy.ndarray numpy complex

    :returns:
        radii, angles
        radii: radius of "radius, angle" representation
        angle: angle of "radius, angle" representation
    :rtype:
        numpy complex or numpy.ndarray numpy complex
    '''
    return np.abs(x), np.angle(x)
