import scipy.signal as sig

from .raw_pipeline import mean, pdf, q

# Peak picking
mean_peaks = mean.map(sig.argrelmax, order=10).pluck(0)
mean_intesnity = (mean_peaks
                  .combine_latest(mean, emit_on=0)
                  .starmap(lambda x, y : y[x]))
q_peak_pos = (mean_peaks
              .combine_latest(q, emit_on=0)
              .starmap(lambda x, y: y[x]))

pdf_peaks = pdf.pluck(1).map(sig.argrelmax, order=10).pluck(0)
pdf_intesnity = (pdf_peaks
                  .combine_latest(pdf, emit_on=0)
                  .starmap(lambda x, y : y[x]))
r_peak_pos = (pdf_peaks
              .combine_latest(r, emit_on=0)
              .starmap(lambda x, y: y[x]))
