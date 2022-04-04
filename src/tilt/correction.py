import math
import numpy as np


def _next_pow2(n):
    return 2**int(math.ceil(math.log(n)/math.log(2.0)))


def _nearest_pow2(x):
    '''
    Get nearest power of 2 for a given input x.

    :param x:
        arbitrary number
    :type x:
        float

    :returns:
        the nearest power of 2 from x
    :rtype:
        float
    '''
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def transfer_function(response, source, dt, smooth):
    '''
    Calculate transfer function and complex coherence between two signals.

    The transfer function is calculated from smoothed cross- and non-smoothed
    autospectral densities of source and response signal. Smoothing is done by
    convolution with a Blackman window.

    The complex transfer function, the autospectral densities, and a
    corresponding frequency vector are returned.

    :param response:
        Sample data of the response signal.
    :type response:
        numpy.ndarray

    :param source:
        Sample data of the source signal.
    :type tr_s:
        numpy.ndarray

    :param dt:
        Sampling interval [s].
    :type dt:
        float

    :param smooth:
        Size of the Blackman window used for smoothing [Hz].
    :type smooth:
        float

    :returns:
        (``freq``, ``XX``, ``YY``, ``Ars``, ``coh``)
        ``freq``: array of frequencies
        ``Grr``: autospectral density of response signal,
        ``Gss``: autospectral density of source signal,
        ``Ars``: source to response transfer function,
        ``coh``: smoothed complex coherence between source and response signal

    :rtype:
        5-:py:class:`tuple` of :py:class:`numpy.ndarray`
    '''

    assert response.size == source.size
    ndat = response.size

    nfft = int(_nearest_pow2(ndat))
    nfft *= 2
    gr = np.zeros(nfft)
    gs = np.zeros(nfft)
    gr[:ndat] = response
    gs[:ndat] = source

    # perform ffts
    Gr = np.fft.rfft(gr)*dt
    Gs = np.fft.rfft(gs)*dt
    freq = np.fft.rfftfreq(nfft, dt)

    # calculate autospectral and crossspectral densities
    Grs = (Gr*Gs.conjugate())
    Grr = (Gr*Gr.conjugate())
    Gss = (Gs*Gs.conjugate())

    nsmooth = int(round(smooth/(freq[1] - freq[0])))
    if nsmooth != 0:
        w = np.blackman(nsmooth)
        Grs_smooth = np.convolve(Grs, w, mode='same')
        Grr_smooth = np.convolve(Grr, w, mode='same')
        Gss_smooth = np.convolve(Gss, w, mode='same')
    else:
        Grs_smooth = Grs
        Grr_smooth = Grr
        Gss_smooth = Gss

    Crs_smooth = Grs_smooth / np.sqrt(Grr_smooth * Gss_smooth)

    # calculate transfer function
    Ars = Crs_smooth * np.sqrt(Grr / Gss)

    return freq, Grr, Gss, Ars, Crs_smooth


def remove_tilt(
        response, source, dt,
        fmin=None,
        fmax=None,
        parallel=True,
        threshold=0.5,
        smooth=1.0,
        g=9.81,
        method='coh',
        trans_coh=None):

    '''
    Remove tilt noise from translational accelerometer recordings.

    See the ``method`` argument for different correction options.
    The correction can optionally be applied only in a selected frequency band.

    The method is described in [BernauerEtAl2020]_.

    :param response:
        Data samples of the accelerometer signal [m/s**2].
    :type response:
        numpy.ndarray

    :param source:
        Data samples of the tilt signal [rad].
    :type source:
        numpy.ndarray

    :param dt:
        Sampling interval [s].
    :type dt:
        float

    :param fmin:
        Minimum frequency for band-limited correction [Hz]. Only applicable in
        ``'coh'`` and ``'freq'`` methods.
    :type fmin:
        :py:class:`float` or ``None``

    :param fmax:
        Maximum frequency for band-limited correction [Hz]. Only applicable in
        ``'coh'`` and ``'freq'`` methods.
    :type fmax:
        :py:class:`float` or ``None``

    :param parallel:
        Flag to indicate if tilt and acceleration axes are parallel (``True``)
        or antiparallel (``False``).
    :type parallel:
        bool

    :param threshold:
        Correction is applied only where ``abs(coherence) >= threshold``. Only
        applicable in ``'coh'`` method.
    :type threshold:
        float

    :param smooth:
        Size of the Blackman window [Hz] used for smoothing when calculating
        the coherence with :py:func:`tilt_utils.transfer_function`. Only
        applicable in ``'coh'`` and ``'freq'`` methods.

    :type smooth:
        float

    :param g:
        Gravitational acceleration [m/s**2].
    :type g:
        float

    :param method:
        Correction method to use. ``'coh'``: apply theoretical transfer
        function where coherence is significant (via frequency domain),
        ``'freq'``: use empirical transfer function estimate (via frequency
        domain), ``'direct'``: apply theoretical transfer function directly (in
        time domain).
    :type method:
        str

    :param trans_coh:
        If given, previously calculated transfer function and complex coherence
        between the tilt and accelerometer signals, used to decide where
        to apply the correction. The size of the given arrays must match the
        size of the spectra of ``source`` and ``response`` (the same
        zero-padding has to be applied). If set to ``None``, it is computed
        from ``response`` and ``source`` using
        :py:func:`tilt_utils.transfer_function`.
    :type trans_coh:
        (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`) or ``None``

    :returns:
        Data samples of corrected accelerometer signal [m/s**2].
    :rtype:
        numpy.ndarray
    '''

    assert response.size == source.size
    assert method in ('direct', 'coh', 'freq')

    sign = 1.0 if parallel else -1.0

    if method == 'direct':
        return response - sign * g * np.sin(source)

    ndat = response.size

    nfft = int(_nearest_pow2(ndat))
    nfft *= 2

    if trans_coh is None:
        Ars, coh = transfer_function(response, source, dt, smooth)[-2:]
    else:
        Ars, coh = trans_coh

    Gr = np.fft.rfft(response, nfft)
    Gs = np.fft.rfft(source, nfft)
    freq = np.fft.rfftfreq(nfft, dt)

    assert Ars.shape == Gr.shape
    assert coh.shape == Gr.shape

    mask = np.where(np.abs(coh) >= threshold, 1.0, 0.0)
    if fmin is not None:
        mask[freq < fmin] = 0.0

    if fmax is not None:
        mask[freq > fmax] = 0.0

    if method == 'coh':
        corr = sign * g * Gs * mask

    elif method == 'freq':
        corr = sign * np.conjugate(Ars) * Gs

    else:
        raise ValueError('Invalid `method` argument: %s' % method)

    return np.fft.irfft(Gr - corr)[:ndat]
