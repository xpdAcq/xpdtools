import scipy.signal as sig
import numpy as np
from .raw_pipeline import mean, pdf, q
from bluesky.callbacks import LivePlot

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


def max_value(pdf):
    return np.amax(pdf.pluck(1))


def total_counts(pdf):
    return len(pdf.pluck(0))


def tallest_peak(pdf):
    peaks = sig.find_peaks(pdf.pluck(1))
    height = []
    r_val = []
    for i in peaks[0]:
        height.append(pdf.pluck(1)[i])
        r_val.append(pdf.pluck(0)[i])
    return np.amax(r_val), np.amax(height)

#doesn't quite work yet
def oscillation_behavior(pdf):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    for i in range(len(pdf.pluck(0))):
        if pdf.pluck(1)[i] >= 20:
            idx = i
            break
    peaks = sig.find_peaks(pdf.pluck(1)[idx:])
    height = []
    r_val = []
    for i in peaks[0]:
        height.append(pdf.pluck(1)[i])
        r_val.append(pdf.pluck(0)[i])

highest_value_stream = pdf.map(max_value)
total_counts_stream = pdf.map(total_counts)
tallest_peak_height, tallest_peak_position = pdf.map(tallest_peak)




