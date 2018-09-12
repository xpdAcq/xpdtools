.. xpdtools documentation master file, created by
   sphinx-quickstart on Thu Dec 28 12:06:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xpdtools's documentation!
====================================

This project holds data processing tools and command line interfaces for
processing data from images to integrated intensities and atomic pair
distribution functions. The tools are put together via a ``streamz`` pipeline.

Installation
============

``xpdtools`` is installable via the ``conda`` package manager.
If you don't have Anaconda (or miniconda) installed please follow the instructions
from `Data Carpentry <https://datacarpentry.org/2016-05-29-PyCon/install.html>`_.

With Anaconda or miniconda please enter into a terminal (on Windows Anaconda
ships with a dedicated command prompt) and type

``conda install xpdtools -c conda-forge``

this will install the xpdtools software and all of its dependencies.

Please check that the software is installed by typing ``image_to_iq -- --help``.
This should display a help description of how to use the ``image_to_iq``
command line interface.

Quickstart
==========
To use the command line interface (CLI) type
``image_to_iq <poni_file> <image_file>``.
For more information about CLI options ``image_to_iq -- --help`` will provide
the help information.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   xpdtools


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
