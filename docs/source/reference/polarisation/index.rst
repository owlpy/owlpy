``polarisation``
================

Tools to perform polarisation analysis on rotational (and other) seismic data.

.. list-table:: Methods
    :widths: 25 75
    :header-rows: 1

    * - Submodule
      - Description
    * - :py:mod:`~owlpy.polarisation.pca`
      - Principal component analysis (PCA) allows you to extract the
        polarisation of the dominant signal in a 2- or 3-component recording.
    * - :py:mod:`~owlpy.polarisation.gridsearch`
      - Get polarization of passing SH/Love waves by exploiting correlation
        between vertical rotational and horizontal acceleration by grid search
        over possible azimuths.

.. toctree::
    :caption: Contents
    
    pca
    gridsearch
