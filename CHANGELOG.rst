===========
 Change Log
===========

.. current developments

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




