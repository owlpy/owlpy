Installation
============

Dependencies
------------

* Python >= 3.6
* setuptools: Use system package.
* NumPy: Use system package.
* Either `ObsPy <https://obspy.org/>`_ or `Pyrocko <https://pyrocko.org/>`_: 
  Install from source, or use pip or conda.

Installation using pip
----------------------

You can install the Pyrots library using pip. Use the ``--no-deps`` option to
prevent pip from installing the dependencies automatically.

.. code-block:: bash

    pip3 install https://github.com/rotational-seismology/pyrots.git --no-deps

Installation from source
------------------------

.. code-block:: bash

   git clone https://github.com/rotational-seismology/pyrots.git
   cd pyrots
   pip3 install . --no-deps
   # or
   python3 setup.py install
