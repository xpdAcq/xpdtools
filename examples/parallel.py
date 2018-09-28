import time

import numpy as np
import pyFAI
import pytest
import tifffile
from distributed import Client
from streamz_ext import destroy_pipeline, Stream
from streamz_ext.link import link
from tornado import gen
from xpdsim import pyfai_poni, image_file
from xpdtools.pipelines.demo_parallel import (
    pipeline_order,
    namespace as g_namespace,
)

img = tifffile.imread(image_file)
geo = pyFAI.load(pyfai_poni)


def raw_pipeline_parallel():
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
    namespace = link(*pipeline_order[:-1], **gg_namespace)

    geo_input = namespace["_geo_input"]
    composition = namespace["_composition"]

    raw_background_dark = namespace["_raw_background_dark"]
    raw_background = namespace["_raw_background"]
    raw_foreground_dark = namespace["_raw_foreground_dark"]
    raw_foreground = namespace["_raw_foreground"]

    print(type(namespace["raw_foreground"]))

    a = namespace['mean']
    futures = a.sink_to_list()
    b = a.buffer(10)
    g = b.gather()
    # g.sink(lambda x: print("gathered data", time.time()))
    LL = g.map(lambda x: time.time()).sink_to_list()
    L = g.sink_to_list()

    a = geo.getPyFAI()
    geo_input.emit(a)
    composition.emit("Au1.0")
    for s in [raw_background_dark, raw_background, raw_foreground_dark]:
        s.emit(np.zeros(img.shape))
    ii = 10
    t0 = time.time()
    for i in range(ii):
        rimg = np.random.random(img.shape)
        raw_foreground.emit(img+rimg)
    while len(L) < ii:
        time.sleep(.01)

    time_diff = [LL[i] - LL[i-1] for i in range(1, ii)]
    print(max(time_diff), min(time_diff), sum(time_diff)/len(time_diff))
    # print([l - min(LL) for l in LL])
    print([l - t0 for l in LL])
    print(max([l - t0 for l in LL])/ii)
    destroy_pipeline(raw_foreground)
    del namespace
    futures.clear()
    L.clear()

raw_pipeline_parallel()
