# xpdtools
[![Build Status](https://travis-ci.org/xpdAcq/xpdtools.svg?branch=master)](https://travis-ci.org/xpdAcq/xpdtools)
[![codecov](https://codecov.io/gh/xpdAcq/xpdtools/branch/master/graph/badge.svg)](https://codecov.io/gh/xpdAcq/xpdtools)
[![Code Health](https://landscape.io/github/xpdAcq/xpdtools/master/landscape.svg?style=flat)](https://landscape.io/github/xpdAcq/xpdtools/master)

Analysis Tools for XPD

Installation
------------
Note that this code is still in beta testing, so things may change. 
Please let me know if you run into any issues.

1. Install [conda](https://conda.io/docs/user-guide/install/index.html)  

1. Add conda-forge to conda:
``conda config add --channels conda-forge``

1. Create a new environment (optional):
``conda create -n xpdtools python=3 pip``

1. If running on windows install pyFAI
``pip install pyfai``

1. Install the build requirements:
``conda install xpdtools``
