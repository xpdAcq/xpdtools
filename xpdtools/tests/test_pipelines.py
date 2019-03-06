import numpy as np
import pyFAI
import pytest
import tifffile

from xpdsim import pyfai_poni, image_file
from xpdtools.pipelines.qoi import (
    max_intensity_mean,
    max_gr_mean,
    pca_pipeline,
    amorphsivity_pipeline)
from xpdtools.pipelines.raw_pipeline import (
    pipeline_order,
    namespace as g_namespace,
)
from rapidz.link import link
from xpdtools.pipelines.extra import z_score_gen, median_gen, std_gen
from xpdtools.pipelines.tomo import (
    tomo_prep,
    tomo_pipeline_piecewise,
    tomo_pipeline_theta,
)
from rapidz import destroy_pipeline, Stream

img = tifffile.imread(image_file)
geo = pyFAI.load(pyfai_poni)


@pytest.mark.parametrize("mask_s", ["first", "none", "auto"])
def test_raw_pipeline(mask_s):
    # link the pipeline up
    namespace = link(*pipeline_order, **g_namespace)

    is_calibration_img = namespace["is_calibration_img"]
    geo_input = namespace["geo_input"]
    img_counter = namespace["img_counter"]
    namespace["mask_setting"]["setting"] = mask_s

    pdf = namespace["pdf"]
    raw_background_dark = namespace["raw_background_dark"]
    raw_background = namespace["raw_background"]
    raw_foreground_dark = namespace["raw_foreground_dark"]
    composition = namespace["composition"]
    raw_foreground = namespace["raw_foreground"]
    sl = pdf.sink_to_list()
    L = namespace["geometry"].sink_to_list()
    ml = namespace["mask"].sink_to_list()

    is_calibration_img.emit(False)
    a = geo.getPyFAI()
    geo_input.emit(a)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    composition.emit("Au")
    img_counter.emit(1)
    raw_foreground.emit(img)
    destroy_pipeline(raw_foreground)
    del namespace
    assert len(L) == 1
    assert ml
    assert len(sl) == 1
    sl.clear()
    L.clear()
    ml.clear()


def test_extra_pipeline():
    # link the pipeline up
    namespace = link(
        *(pipeline_order + [median_gen, std_gen, z_score_gen]), **g_namespace
    )

    geometry = namespace["geometry"]

    z_score = namespace["z_score"]
    raw_background_dark = namespace["raw_background_dark"]
    raw_background = namespace["raw_background"]
    raw_foreground_dark = namespace["raw_foreground_dark"]
    raw_foreground = namespace["raw_foreground"]

    sl = z_score.sink_to_list()
    geometry.emit(geo)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    raw_foreground.emit(img)
    del namespace
    destroy_pipeline(raw_foreground)
    assert len(sl) == 1
    sl.clear()


def test_qoi_pipeline():
    # link the pipeline up
    namespace = link(
        *(pipeline_order + [max_intensity_mean, max_gr_mean]), **g_namespace
    )

    geometry = namespace["geometry"]

    mean_max = namespace["mean_max"]
    raw_background_dark = namespace["raw_background_dark"]
    raw_background = namespace["raw_background"]
    raw_foreground_dark = namespace["raw_foreground_dark"]
    raw_foreground = namespace["raw_foreground"]

    sl = mean_max.sink_to_list()
    geometry.emit(geo)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    raw_foreground.emit(img)
    del namespace
    destroy_pipeline(raw_foreground)
    assert len(sl) == 1
    sl.clear()


def test_tomo_piecewise_pipeline():
    ns = dict(
        qoi=Stream(),
        x=Stream(),
        th=Stream(),
        th_dim=Stream(),
        x_dim=Stream(),
        th_extents=Stream(),
        x_extents=Stream(),
        center=Stream(),
    )
    x_linspace = np.linspace(0, 5, 6)
    th_linspace = np.linspace(0, 180, 6)

    ns["th_dimension"] = len(th_linspace)
    ns["x_dimension"] = len(x_linspace)

    ns.update(**link(*[tomo_prep, tomo_pipeline_piecewise], **ns))

    L = ns["rec"].sink_to_list()

    ns["th_dim"].emit(len(th_linspace))
    ns["x_dim"].emit(len(x_linspace))
    ns["th_extents"].emit([0, 180])
    ns["x_extents"].emit([x_linspace[0], x_linspace[-1]])
    ns["center"].emit(2.5)

    # np.random.seed(42)

    for x in x_linspace:
        for th in th_linspace:
            ns["x"].emit(x)
            ns["th"].emit(th)
            ns["qoi"].emit(np.random.random())

    assert len(L) == len(x_linspace) * len(th_linspace)
    assert L[-1].shape == (len(x_linspace), len(th_linspace))

    destroy_pipeline(ns["qoi"])
    del ns
    L.clear()


def test_tomo_pipeline_theta():
    ns = dict(qoi=Stream(), theta=Stream(), center=Stream())

    ns.update(tomo_pipeline_theta(**ns))
    L = ns["rec"].sink_to_list()
    # np.random.seed(42)

    th_linspace = np.linspace(0, 180, 6)
    ns["center"].emit(3)

    for th in th_linspace:
        ns["theta"].emit(th)
        ns["qoi"].emit(np.random.random((6, 6)))
    assert len(L) == 6
    assert L[-1].shape == (6, 6, 6)
    destroy_pipeline(ns["qoi"])
    del ns
    L.clear()


def test_pca_pipeline():
    ns = dict(data=Stream(), start=Stream())

    ns.update(pca_pipeline(**ns))
    L = ns["scores"].sink_to_list()

    # np.random.seed(42)
    for i in range(10):
        a = np.zeros(10)
        a[i] = 1
        ns["data"].emit(a)

    assert len(L) == 10
    assert L[-1].shape == (10, 9)
    destroy_pipeline(ns["data"])
    del ns
    L.clear()


def test_amorphous_pipeline():
    pdf = Stream()
    ns = amorphsivity_pipeline(pdf)
    L = ns['amorphsivity'].sink_to_list()
    a = np.ones(10)
    pdf.emit(a)
    assert L[0] == np.sum(a[6:])
