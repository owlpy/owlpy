Tilt correction
===============

This script implements a simple example for dynamic tilt correction. It
reproduces the tilt table experiment from [BernauerEtAl2020]_ based on the
method by [CrawfordAndWebb2000]_.

The example demonstrates the different correction variants provided by
:py:func:`owlpy.tilt.correction.remove_tilt`.

The example data required by this script is included in the OwlPy example data
collection. It can be downloaded with the script :download:`get_example_data.sh
</../../examples/get_example_data.sh>`.

.. figure :: /_static/tilt_correction_step_table.png
    :align: center
    :alt: output of tilt_correction_step_table.py

.. literalinclude :: /../../examples/tilt_correction_step_table.py
    :caption: :download:`tilt_correction_step_table.py </../../examples/tilt_correction_step_table.py>`
    :language: python

