import numpy as np
import tomopy


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
    esa[thp, 0, xp] = q
    return esa


def tomo_pipeline_theta(qoi, theta, center, algorithm="gridrec", **kwargs):
    rec = (
        qoi.map(reshape, stream_name="reshape")
        .map(tomopy.minus_log)
        .zip(theta)
        .accumulate(append_data)
        .combine_latest(center, emit_on=0)
        .map(flatten)
        .starmap(tomopy.recon, algorithm=algorithm)
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
    **kwargs
):
    """Perform a tomographic reconstruction on a QOI"""
    a = qoi.zip(th_pos, x_pos)
    sineogram = a.accumulate(fill_sinogram)
    # This is created at the start document and bypasses the fill_sinogram
    # function
    # TODO: make a function for the np.ones
    th_dim.zip(x_dim).starmap(lambda th, x: np.ones((th, 1, x))).connect(
        sineogram
    )

    rec = (
        sineogram.map(np.nan_to_num)
        .map(tomopy.minus_log)
        .map(np.nan_to_num)
        .combine_latest(th_ext, center, emit_on=0)
        .starmap(tomopy.recon, algorithm=algorithm)
        .map(np.squeeze)
    )
    return locals()
