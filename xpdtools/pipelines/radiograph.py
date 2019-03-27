"""Pipeline chunks for handling radiographs (full field tomography
preprocessing)"""
from rapidz import Stream, identity
import operator as op
from tomopy import normalize


def uniqueish(state, new):
    if state is False:
        return new, False
    return new, state != new


def sum_state(state, new):
    return state + 1


def unique_data(data, **kwargs):
    """Similar to ``rapidz.unique`` but doesn't emit for the first element
    of the stream

    Parameters
    ----------
    data : Stream
        The stream of data
    kwargs

    Returns
    -------

    """
    unique = data.accumulate(uniqueish, returns_state=True,
                             start=False).filter(bool)
    return locals()


def average(data: Stream, reset: Stream = None, **kwargs):
    """Perform a running average

    Parameters
    ----------
    data : Stream
        The data to be averaged as a stream
    reset : Stream, optional
        If provided, when data comes from this stream reset the averaging

    Returns
    -------
    locals : dict
        The locals

    """
    img_sum = data.accumulate(op.add, reset_stream=reset)
    img_count = data.accumulate(sum_state,
                                start=0,
                                reset_stream=reset)
    ave_img = img_sum.zip(img_count).starmap(op.truediv)

    return locals()


def radiograph_correction(img: Stream, dark: Stream, flat_field: Stream,
                          motors: Stream, **kwargs):
    norm_img = img.combine_latest(flat_field, dark, emit_on=0).starmap(
        normalize)

    return locals()
