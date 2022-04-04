``tilt``
========

.. list-table:: Methods
    :widths: 25 75
    :header-rows: 1

    * - Submodule
      - Description
    * - :py:mod:`~owlpy.tilt.correction`
      - Remove tilt-induced accelerations from horizontal seismogram recordings
        using collocated rotational recordings. Options include the possibility
        to apply the correction in a data adaptive approach, only where the
        coherence between rotational and acceleration signal is high. Such an
        approach is useful to prevent pollution of the corrected signal with
        noise from the rotational recordings.

.. toctree::
    :caption: Contents
    
    correction
