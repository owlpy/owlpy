
from setuptools import setup

setup(
    name='owlpy',
    version='0.0.1',
    author='The OwlPy Developers',
    maintainer='Sebastian Heimann',
    maintainer_email='sebastian.heimann@uni-potsdam.de',
    license='AGPLv3',
    package_dir={
        'owlpy': 'src'
    },
    packages=[
        'owlpy',
        'owlpy.polarisation',
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
