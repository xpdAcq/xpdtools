import numpy as np
from streamz_ext import Stream
from xpdtools.tools import sq_getter, fq_getter, pdf_getter


def make_pipeline():
    """Make pipeline for S(Q), F(Q) and PDF generation"""
    mean = Stream()
    q = Stream()
    # PDF
    composition = Stream(stream_name='composition')
    iq_comp = (
        q.combine_latest(mean, emit_on=1)
        .combine_latest(composition, emit_on=0))
    iq_comp_map = (iq_comp.map(lambda x: (x[0][0], x[0][1], x[1])))

    # TODO: split these all up into their components ((r, pdf), (q, fq)...)
    sq = iq_comp_map.starmap(sq_getter, stream_name='sq', **(
        dict(dataformat='QA', qmaxinst=28, qmax=25, rstep=np.pi / 25)))
    fq = iq_comp_map.starmap(fq_getter, stream_name='fq', **(
        dict(dataformat='QA', qmaxinst=28, qmax=25, rstep=np.pi / 25)))
    pdf = iq_comp_map.starmap(pdf_getter, stream_name='pdf', **(
        dict(dataformat='QA', qmaxinst=28, qmax=22, rstep=np.pi / 22)))
    return {
        'mean': mean,
        'q': q,
        'composition': composition,
        'iq_comp': iq_comp,
        'iq_comp_map': iq_comp_map,
        'sq': sq,
        'fq': fq,
        'pdf': pdf
    }
