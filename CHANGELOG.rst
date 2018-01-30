===========
 Change Log
===========

.. current developments

v0.1.6
====================

**Added:**

* Quickstart to ``Readme.md``


**Changed:**

* Save z score as ``.tif`` file

* ``binned_outlier`` now uses input mask (if any) to remove pixels before
  running the binned outlier algorithm.


**Fixed:**

* All integrated values are processed with ``np.nan_to_num`` before output.




v0.1.5
====================

**Added:**

* Kwarg for flipping the input mask (may be needed for fit2d masks)


**Removed:**

* Docs for beamstop mask


**Fixed:**

* Polarization works properly

* Multi image works properly
* Code health badge

* Docs for ``mask_img`` ``alpha``




v0.1.4
====================

**Fixed:**

* removed relative import from CLI




v0.1.3
====================

**Added:**

* Test of the CLI (to make sure it writes out files now)

* Tests of many (although not all) of the tools.

* Added support for ``scikit-beam=0.0.12`` which lacks som cached data


**Changed:**

* Readme now reflects the conda package

* Travis now has a display




v0.1.2
====================

**Added:**

* Dedicated XPD pipeline which has the capacity to only mask the first 
  image in a series.




v0.1.1
====================

**Added:**

* Benchmark scripts for speed testing (Note that these run on local files 
  currently)
* Numba for median masking, giving a speedup


**Changed:**

* Most ``zip_latest`` nodes have been changed to ``combine_latest`` to avoid 
  unwanted buffering.
* Use ``BinnedStatistics`D`` properties for masking, which reduces recomputation


**Removed:**

* ``streamz`` dep, now the project depends on ``streamz_ext``




v0.1.0
====================

**Added:**

* Command Line interface for integration
* Add rever changelog activity
* Speed up masking via median based sigma clipping
* Z score visualization to callback pipeline


**Changed:**

* Fixed up main pipeline




