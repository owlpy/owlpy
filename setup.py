
from setuptools import setup

setup(
    name='pyrots',
    version='0.0.1',
    author='The Pyrots Developers',
    maintainer='Sebastian Heimann',
    maintainer_email='sebastian.heimann@uni-potsdam.de',
    license='AGPLv3',
    package_dir={
        'pyrots': 'src'
    },
    packages=[
        'pyrots',
        'pyrots.polarisation',
    ],
    install_requires=[
        'numpy',
        # 'scipy',
    ],
    extras_require={
        'obspy_compatibility': ['obspy'],
        'pyrocko_compatibility': ['pyrocko'],
    },
)
