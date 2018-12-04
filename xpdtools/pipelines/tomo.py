import numpy as np
import tomopy
import operator as op


def append_data(acc, pt):
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


def tomo_pipeline_theta(qoi, theta, center, algorithm="gridrec"):
    tomo_node = (
        qoi.map(reshape, stream_name="reshape")
        .map(tomopy.minus_log)
        .zip(theta)
        .accumulate(append_data)
        .combine_latest(center, emit_on=0)
        .map(flatten)
        .starmap(tomopy.recon, algorithm=algorithm)
    )
    return locals()


def tomo_pipeline_x(qoi, y, center, algorithm="gridrec"):
    tomo_node = (
        qoi.map(lambda x: np.reshape(x, (x[0], 1, len(theta))))
        .map(tomopy.minus_log)
        .zip(y)
        .accumulate(append_data)
        .combine_latest(center, emit_on=0)
        .map(lambda x: (*x[0], x[1]))
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
    x_pos = x.combine_latest(x_ext, emit_on=x).starmap(min_pos)
    th_pos = th.combine_latest(th_ext, emit_on=th).starmap(min_pos)
    return locals()


# TODO: this might not be ok long term, since many things will want to access
#  this array and that might not be stable, maybe need to make new copies
#  maybe better as an accumulator which mints new arrays?
def fill_sinogram(q, xp, thp, esa):
    esa[xp, 0, thp] = q
    return esa


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
    empty_sineogram_array = th_dim.zip(x_dim).starmap(
        lambda a, b: np.ones((a, 1, b))
    )
    sineogram = (
        qoi.zip(th_pos, x_pos)
        .combine_latest(empty_sineogram_array, emit_on=0)
        .map(flatten)
        .starmap(fill_sinogram)
    )

    rec = (
        sineogram.map(tomopy.minus_log)
        .combine_latest(th_ext, center, emit_on=0)
        .starmap(tomopy.recon, algorithm=algorithm)
    )
    return locals()
