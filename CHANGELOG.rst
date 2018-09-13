===========
 Change Log
===========

.. current developments

v0.4.3
====================

**Fixed:**

* Fix bug where ``is_calibration_img`` was not being checked properly
* no mask setting, we don't need kwargs for no mask




v0.4.2
====================

**Added:**

* ``calib_setting`` dict to the raw pipeline, if
  ``calib_setting['setting'] is False`` then the calibration gui will not be
  run even for calibration runs


**Changed:**

* ``pyFAI`` imports for calibration are now inside the functions so we don't
  run the imports until they are actually needed.




v0.4.1
====================

**Changed:**

* Added ``pool`` to `xpdtools.tools.binned_outlier`` and
  ``xpdtools.tools.mask_img`` for an executor pool interface
* Added ``tqdm`` for slow mean masking


**Fixed:**

* ``binned_outlier`` properly uses existing masks




v0.4.0
====================

**Added:**

* Pipeline and tools for flatfield correction measurements


**Changed:**

* Added default mask kwargs to ``mask_kwargs`` for usability




v0.3.1
====================

**Fixed:**

* ``starmap`` into ``img_calibration`` rather than just ``map`` which gives
  correct alignment between the inputs and expected arguments




v0.3.0
====================

**Added:**

* Quantity of interest support


**Changed:**

* tth reported in degrees
* Run release before conda forge so we use the self generated tarballs


**Fixed:**

* Don't break API with ``generate_binner``




v0.2.0
====================

**Added:**

* ``pipelines.extra`` module which holds extra nodes (zscore, median, etc)

* numba compiled ``zscore`` for faster zscore computation


**Changed:**

* removed zscore, median, and std from the base pipeline

* use ``map`` rather than for loop for zscore


**Removed:**

* ``xpd_raw_pipeline`` module




v0.1.9
====================

**Changed:**

* Merged xpd and standard pipelines into one pipeline

* Exposed the mask, fq, and pdf kwargs to the user better.
  Now the kwarg dicts are from the nodes and can be updated.


**Deprecated:**

* xpd pipeline (it is now in the standard pipeline)


**Fixed:**

* ``iq_comp`` now is combined via a ``combine_latest`` rather than a zip




v0.1.8
====================



v0.1.7
====================

**Changed:**

* Zscore is now turned into ``float16`` before saving to reduce size on disk


**Fixed:**

* Command line interface destroys sinks so it shouldn't blow up memory

* ``generate_binner`` now has max q of the max q




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




