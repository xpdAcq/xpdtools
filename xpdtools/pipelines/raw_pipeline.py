"""Main pipeline for processing images to I(Q) and PDF"""

from streamz_ext.link import link
from xpdtools.pipelines.calibration import make_pipeline as cal_mp
from xpdtools.pipelines.image_preprocess import make_pipeline as ipmp
from xpdtools.pipelines.integration import make_pipeline as bp
from xpdtools.pipelines.masking import make_pipeline as mp, mask_setting  # noqa: F401
from xpdtools.pipelines.pdf import make_pipeline as pdfp
from xpdtools.pipelines.scattering_correction import make_pipeline as scat_cor


def make_pipeline():
    # The process of making a pipeline is gluing pieces together
    pipeline_dict = link(*[ipmp(), cal_mp(), scat_cor(), mp(), bp(), pdfp()])
    return pipeline_dict
