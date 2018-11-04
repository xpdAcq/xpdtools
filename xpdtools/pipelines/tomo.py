import numpy as np
import tomopy


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


def tomo_pipeline_theta(qoi, theta, center, algorithm="gridrec"):
    tomo_node = (
        qoi.map(reshape,
                stream_name='reshape')
        .map(tomopy.minus_log)
        .zip(theta)
        .accumulate(append_data)
        .combine_latest(center, emit_on=0)
        .map(flatten
    )
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
