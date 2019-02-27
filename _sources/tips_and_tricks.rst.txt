Tips and Tricks
===============
Here is a collection of tips and tricks for using xpdtools.

Weak Scattering and Air Scattering
----------------------------------
Weak scattering data benefits from image to image background subtraction.
Often for weak scattering materials the air scattering of the beamline becomes
non trivial.
At this point we need to remove the air scattering (although minimizing it on
the experimental side is preferable).
After the experiment is done one way to remove air scattering (and its shadows)
is to perform background subtraction on the images.
Since the air scattering is non azimuthally symmetric simple integration and 
subtraction of the integrated background will not work, as the integration
has removed any azimuthal information from the data.
``xpdtools`` provides a flag, ``--bg_file``, to have a background image
subtracted from the foreground image.
The effect of this can be seen in the standard deviation and the z score.


Median vs. Mean masking
-----------------------
The automasking code has a few flags to modify its behavior, one of the most
important is ``--auto_type``.
This flag governs which algorithm is used to create the statistical mask.
The default is ``median`` which does a single pass for each azimuthal ring
and finds outliers.
The flag can also be set to ``mean`` which computes the mean for each ring
and removes a single pixel repeatidely until the all the outliers with
a z score greater than ``alpha`` have been removed.
The ``mean`` method is much slower, but is generally more accurate.
