import logging
import time

import numpy as np
import pyFAI
import pytest
import tifffile
from tornado import gen

from xpdsim import pyfai_poni, image_file
from xpdtools.pipelines.demo_parallel import (
    pipeline_order,
    namespace as g_namespace,
)
from rapidz.link import link
from rapidz import destroy_pipeline, Stream

img = tifffile.imread(image_file)
geo = pyFAI.load(pyfai_poni)


gen_test = pytest.mark.gen_test

@gen_test(timeout=10)
@pytest.mark.parametrize("n", [
    'bg_corrected_img',
    'mask', 'binner', 'mean',
])
def test_raw_pipeline_parallel(n):
    # caplog.set_level(logging.CRITICAL)
    # link the pipeline up
    gg_namespace = dict(g_namespace)
    s_ns = {
        k: v.scatter(backend="thread")
        for k, v in gg_namespace.items()
        if isinstance(v, Stream)
    }
    gg_namespace.update(
        {"_" + k: v for k, v in gg_namespace.items() if isinstance(v, Stream)}
    )
    gg_namespace.update(s_ns)
    namespace = link(*pipeline_order, **gg_namespace)

    geo_input = namespace["_geo_input"]

    raw_background_dark = namespace["_raw_background_dark"]
    raw_background = namespace["_raw_background"]
    raw_foreground_dark = namespace["_raw_foreground_dark"]
    raw_foreground = namespace["_raw_foreground"]

    a = namespace[n]
    futures = a.sink_to_list()
    b = a.buffer(10)
    g = b.gather()
    g.sink(lambda x: print("gathered data", time.time()))
    L = g.sink_to_list()

    a = geo.getPyFAI()
    yield geo_input.emit(a)
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        yield s.emit(np.zeros(img.shape))
    ii = 2
    for i in range(ii):
        rimg = np.random.random(img.shape)
        yield raw_foreground.emit(img+rimg)
    while len(L) < ii:
        yield gen.sleep(.01)

    destroy_pipeline(raw_foreground)
    del namespace
    futures.clear()
    L.clear()
