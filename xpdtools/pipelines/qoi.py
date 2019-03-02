import numpy as np

from toolz import get
from xpdtools.tools import decomp
import operator as op


def max_intensity_mean(mean, q, **kwargs):
    q_at_mean_max = (
        mean.map(np.nanargmax).combine_latest(q, emit_on=0).starmap(get)
    )
    mean_max = mean.map(np.nanmax)
    return locals()


def max_gr_mean(pdf, **kwargs):
    r = pdf.pluck(0)
    gr = pdf.pluck(1)
    r_at_gr_max = (
        gr.map(np.nanargmax).combine_latest(r, emit_on=0).starmap(get)
    )
    gr_max = gr.map(np.nanmax)
    return locals()


def pca_pipeline(data, start, n_components=.9, **kwargs):
    concat_data = data.accumulate(lambda acc, x: acc + [x], start=[])
    start.sink(lambda x: concat_data.state.clear())
    pca = concat_data.map(decomp, n_components=n_components, **kwargs)
    components = pca.pluck(0)
    scores = pca.pluck(1)
    return locals()


def amorphsivity_pipeline(pdf, lower_bound_percentage=.66):
    """Compute the amorphsivity of the material via the G(r).

    This computes a proxy of how amorphous the material is by comparing the
    integrated norm probability of finding an interatomic distance at high
    distance values against the highest peak in the PDF. Crystalline samples
    will have peaks at high distances, as the probability of finding an atom
    at a particular distance is highly localized to certain distances.
    Amorphous samples have fewer peaks with lower intensity at high distances,
    at those interatomic distances the probability of finding an atom is
    roughly the same as any other distance.

    Parameters
    ----------
    pdf : Stream
        The stream of PDFs
    lower_bound_percentage : float, optional
        The fraction of the PDF used to define the lower bound of high
        distances. .9 would include only the last few data points, while .5
        would include half the PDF, defaults to .66

    Returns
    -------
    locals : dict
        The locals
    """
    gr = pdf.pluck(1)
    # Get the lower bound index
    index = gr.map(len).map(op.mul, lower_bound_percentage).map(int)
    abs_max = gr.map(np.abs).map(np.max)
    integrated_max = (
        gr.combine_latest(index, emit_on=0)
        .starmap(lambda x, y: x[y:])
        .map(np.abs)
        .map(np.sum)
    )
    amorphsivity = integrated_max.zip(abs_max).starmap(op.truediv)
    return locals()


"""
r = pdf.pluck(0)
true_pdf = pdf.pluck(1)

# Peak picking
pmp = mean.map(sig.argrelmax, order=20)
mean_peaks = pmp.pluck(0)
mean_intensity = mean_peaks.combine_latest(mean, emit_on=0).starmap(
    lambda x, y: y[x]
)
q_peak_pos = mean_peaks.combine_latest(q, emit_on=0).starmap(lambda x, y: y[x])
ppp = true_pdf.map(sig.argrelmax, order=5)
pdf_peaks = ppp.pluck(0)
pdf_intensity = pdf_peaks.combine_latest(true_pdf, emit_on=0).starmap(
    lambda x, y: y[x]
)
r_peak_pos = pdf_peaks.combine_latest(r, emit_on=0).starmap(lambda x, y: y[x])

# Hack to properly order the pipeline
for s, t in zip([mean, true_pdf], [pmp, ppp]):
    for n in s.downstreams.data:
        if n() is t:
            break
    s.downstreams.data._od.move_to_end(n, last=True)
    del n

pdf_argrelmax_kwargs = pdf_peaks.upstreams[0].kwargs
mean_argrelmax_kwargs = mean_peaks.upstreams[0].kwargs
"""
