import numpy as np
import tomopy
from rapidz import no_default


def recon_wrapper(projection, theta, center, **kwargs):
    """A wrapper around ``tomopy.recon`` which properly adds or loops over
    dimensions to make the tomographic reconstruction work.

    Parameters
    ----------
    projection : ndarray
        The sinogram of the projection data. The first axis is theta, the last
        is x. If additional axes are included these will be looped over and
        put back to gether
    theta : ndarray
        The theta values
    center : float
        The rotation axis of the sample in pixels

    Returns
    -------
    out : ndarray
        The reconstructed data
        The output dimension will match the input dimension
    """
    shape = projection.shape
    # This is a measurement of scalars (put onto a th, x grid)
    if len(shape) == 2:
        data = np.expand_dims(projection, axis=1)
        out = tomopy.recon(data, theta, center, **kwargs)
    # This is a measurement of images (in full field, [th, y, x]) or a
    # measurment of vectors (in pencil, [th, v_dim, x])
    elif len(shape) == 3:
        out = tomopy.recon(projection, theta, center, **kwargs)
    # This is a measurement of diffraction images (2D, 4D total)
    elif len(shape) == 4:
        outs = []
        # (th, img_i, img_j, x)
        # Parallelize this?
        for i in range(shape[2]):
            outs.append(
                tomopy.recon(projection[:, :, i, :], theta, center, **kwargs)
            )
        outs2 = [np.expand_dims(o, axis=2) for o in outs]
        out = np.concatenate(outs2, axis=2)
    else:
        raise RuntimeError(
            f"There is not a reconstruction system setup for"
            "a {len(shape)} array"
        )
    return np.squeeze(out)


def append_data(acc, pt):
    """Append data to array for full field tomo

    Parameters
    ----------
    acc : tuple
        Accumulated projection and theta data
        ``(accumulated_projection, accumulated_theta)``
    pt : tuple
        The projection and theta data ``(new_projection, new_theta)``

    Returns
    -------
    ap : ndarray
        The accumulated projection
    at : ndarray
        The accumuated theta points
    """
    p, t = pt
    ap, at = acc
    ap = np.concatenate((ap, p))
    at = np.append(at, t)
    return ap, at


def flatten(x):
    return (*x[0], x[1])


def reshape(x):
    return np.reshape(x, (1, *x.shape))


def min_pos(x, y):
    return np.argmin((x - y) ** 2)


# TODO: this might not be ok long term, since many things will want to access
#  this array and that might not be stable, maybe need to make new copies
#  maybe better as an accumulator which mints new arrays?
def fill_sinogram(esa, q_thp_xp):
    """

    Parameters
    ----------
    esa : np.array
        The empty sinogram array
    q_thp_xp: tuple
        The theta position, x position, and QOI in a tuple

    Returns
    -------

    """
    q, thp, xp = q_thp_xp
    # Copy the array so we have independent access to it
    # esa = esa.copy()
    esa[thp, xp] = q
    return esa


def conditional_squeeze(arr, axis):
    shape = arr.shape
    if shape[axis] == 1:
        return np.squeeze(arr, axis)
    else:
        return arr


# TODO: unify the reconstruction section of the pipeline and make the prep
#  produce the sinograms

def sort_sinogram(sinogram, theta):
    """Sort a sinogram by its theta values for easy viewing

    Parameters
    ----------
    sinogram : ndarray
        The sinogram, theta must be the first axis
    theta : ndarray
        The theta values

    Returns
    -------
    ndarray :
        The sorted array

    """
    return sinogram[theta.argsort()[::-1]]


def tomo_pipeline_theta(qoi, theta, center, algorithm="gridrec", **kwargs):
    sinogram_theta = (
        # replace with expand_dims
        qoi.map(reshape, stream_name="reshape")
        .map(tomopy.minus_log)
        .zip(theta)
        .accumulate(append_data)
    )
    sinogram = (
        sinogram_theta
        # Sort the sinogram by theta for out of order scans
        .starmap(sort_sinogram)
        # The sinogram accumulates over theta so we can squeeze
        .map(conditional_squeeze, 1)
    )
    sinogram.sink(print)
    rec = (
        sinogram_theta.combine_latest(center, emit_on=0)
        .map(flatten)
        .starmap(recon_wrapper, algorithm=algorithm)
    )
    return locals()


def tomo_prep(x, th, th_dim, x_dim, th_extents, x_extents, **kwargs):
    """Preperation chunk for tomography, mostly munging positional data
       into array coordinates"""
    # dims -> (91, 44)
    # extents -> ([180, 0], [-8, 9.5])
    x_ext = x_extents.zip(x_dim).map(flatten).starmap(np.linspace)
    th_ext = (
        th_extents.zip(th_dim)
        .map(flatten)
        .starmap(np.linspace)
        .map(np.deg2rad)
    )
    x_pos = x.combine_latest(x_ext, emit_on=0).starmap(min_pos)
    th_pos = (
        th.map(np.deg2rad).combine_latest(th_ext, emit_on=0).starmap(min_pos)
    )
    return locals()


def tomo_pipeline_piecewise(
    qoi,
    x_pos,
    th_pos,
    th_dim,
    x_dim,
    center,
    th_ext,
    algorithm="gridrec",
    **kwargs,
):
    """Perform a tomographic reconstruction on a QOI"""
    a = qoi.zip(th_pos, x_pos)
    sinogram = a.accumulate(fill_sinogram)
    # This is created at the start document and bypasses the fill_sinogram
    # function
    # TODO: make a function for the np.ones
    th_dim.zip(x_dim).starmap(lambda th, x: np.ones((th, x))).sink(
        lambda x: setattr(sinogram, "state", x)
    )

    rec = (
        sinogram.map(np.nan_to_num)
        .map(tomopy.minus_log)
        .map(np.nan_to_num)
        .combine_latest(th_ext, center, emit_on=0)
        .starmap(recon_wrapper, algorithm=algorithm)
    )
    return locals()


def acc(old, new):
    oldtomo, oldz = old
    newtomo, newz = new
    if oldz == newz:
        # XXX: make sure that oldtomo is a 3D array, pad otherwise
        oldtomo[..., -1] = np.squeeze(newtomo)
    else:
        # XXX: we might need to use a slightly different function
        oldtomo = np.dstack((oldtomo, newtomo))
    return oldtomo, newz


def tomo_stack_2D(rec, stack_position, start, **kwargs):
    rec_3D = (
        rec.map(np.atleast_3d)
        .combine_latest(stack_position, emit_on=0)
        .accumulate(acc)
        .pluck(0)
    )
    start.sink(lambda x: setattr(rec_3D, "start", no_default))
    return locals()
