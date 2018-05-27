import scipy.signal as sig

from .raw_pipeline import mean, pdf, q

r = pdf.pluck(0)
true_pdf = pdf.pluck(1)

# Peak picking
mean_peaks = mean.map(sig.argrelmax, order=20).pluck(0)
mean_intensity = (mean_peaks
                  .combine_latest(mean, emit_on=0)
                  .starmap(lambda x, y: y[x]))
q_peak_pos = (mean_peaks
              .combine_latest(q, emit_on=0)
              .starmap(lambda x, y: y[x]))

pdf_peaks = true_pdf.map(sig.argrelmax, order=10).pluck(0)
pdf_intensity = (pdf_peaks
                 .combine_latest(true_pdf, emit_on=0)
                 .starmap(lambda x, y: y[x]))
r_peak_pos = (pdf_peaks
              .combine_latest(r, emit_on=0)
              .starmap(lambda x, y: y[x]))

# Hack to properly order the pipeline
for s, t in zip([mean, true_pdf], [mean_peaks, pdf_peaks]):
    for n in s.downstreams.data:
        if n() is t.upstreams[0]:
            break
    s.downstreams.data._od.move_to_end(n, last=True)
    del n

pdf_argrelmax_kwargs = pdf_peaks.upstreams[0].kwargs
mean_argrelmax_kwargs = mean_peaks.upstreams[0].kwargs
