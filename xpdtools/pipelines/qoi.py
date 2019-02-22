import numpy as np

from toolz import get
from xpdtools.tools import decomp


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


def pca_pipeline(data, start, variance=.9, **kwargs):
    concat_data = data.accumulate(lambda acc, x: acc.append(x), start=[])
    start.sink(lambda x: concat_data.state.clear())
    pca = concat_data.map(decomp, variance=variance, **kwargs)
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
