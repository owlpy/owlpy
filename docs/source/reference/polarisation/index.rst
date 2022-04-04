``polarisation``
================

Tools to perform polarisation analysis on rotational (and other) seismic data.

.. list-table:: Methods
    :widths: 25 75
    :header-rows: 1

    * - Submodule
      - Description
    * - :py:mod:`~owlpy.polarisation.pca`
      - Extract polarisation of the dominant signal in a 2- or 3-component
        seismic recording using principal component analysis (PCA).
    * - :py:mod:`~owlpy.polarisation.gridsearch`
      - Get polarization of passing SH/Love waves through exploitation of the
        correlation between vertical rotational and horizontal acceleration by
        grid search over possible azimuths.

.. toctree::
    :caption: Contents

    pca
    gridsearch
