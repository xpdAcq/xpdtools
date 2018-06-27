"""Main pipeline for processing images to I(Q) and PDF"""

from streamz_ext.link import link
from xpdtools.pipelines.calibration import cali
from xpdtools.pipelines.image_preprocess import make_pipeline as ipmp
from xpdtools.pipelines.integration import binner_pipeline as bp
from xpdtools.pipelines.masking import make_pipeline as mp, mask_setting
from xpdtools.pipelines.pdf import make_pipeline as pdfp
from xpdtools.pipelines.scattering_correction import make_pipeline as scat_cor


def make_pipeline():
    # The process of making a pipeline is gluing pieces together
    pipeline_dict = link(
        *[
            ipmp(),
            cali(),
            scat_cor(),
            mp(),
            bp(),
            pdfp()
        ]
    )
    return pipeline_dict


pipeline = make_pipeline()
print(pipeline)
# Tie all the kwargs together (so changes in one node change the rest)
mask_kwargs = pipeline['all_mask'].kwargs

fq_kwargs = pipeline['fq'].kwargs
pipeline['sq'].kwargs = fq_kwargs
pdf_kwargs = pipeline['pdf'].kwargs
