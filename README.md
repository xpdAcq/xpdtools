# xpdtools
[![Build Status](https://travis-ci.org/xpdAcq/xpdtools.svg?branch=master)](https://travis-ci.org/xpdAcq/xpdtools)
[![codecov](https://codecov.io/gh/xpdAcq/xpdtools/branch/master/graph/badge.svg)](https://codecov.io/gh/xpdAcq/xpdtools)
[![Code Health](https://landscape.io/github/xpdAcq/xpdAn/master/landscape.svg?style=flat)](https://landscape.io/github/xpdAcq/xpdtools/master)

Analysis Tools for XPD

Installation
------------
Note that this code is still in beta testing, so things may change. Please
let me know i you run into any issues.

Add conda-forge: 
``conda config add --channels conda-forge``

Create a new environment (optional):
``conda create -n xpdtools python=3 pip``

Install the build requirements:
``conda install --file requirements/build.txt``

Install the run requirements:
``conda install -file requirements/run.txt``

Install streamz:
``pip install -r requirements/pip.txt``

If on windows also run:
``pip install pyfai``

Finally run:
``pip install git+https://github.com/xpdAcq/xpdtools.git#egg=xpdtools``