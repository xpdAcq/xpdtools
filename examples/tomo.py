import numpy as np
import tomopy
import dxchange
import os

from rapidz import Stream
from bluesky.callbacks.broker import LiveImage
import matplotlib.pyplot as plt
from bluesky.utils import install_qt_kicker
from xpdtools.pipelines.tomo import tomo_pipeline_theta


fname = os.path.expanduser('~/Downloads/tooth.h5')

start = 0
end = 1
proj, flat, dark, theta = dxchange.read_aps_32id(fname, sino=(start, end))
proj = tomopy.normalize(proj, flat, dark)

rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)


backend = 'thread'

center = Stream(stream_name='center')
proj_node = Stream(stream_name='projection')
theta_node = Stream(stream_name='theta')

# z.sink(lambda x: print(x[0].shape))
install_qt_kicker()

li = LiveImage('hi', cmap='viridis')
ns = tomo_pipeline_theta((proj_node
                          # .scatter(backend=backend)
                          ),
                         (theta_node
                          # .scatter(backend=backend)
                          ),
                    center=(center
                            # .scatter(backend=backend)
                            ))
z = ns['tomo_node']
zz = (z
      # .buffer(1000).gather()
      .pluck(0))
# zz.sink(print)
zz.sink(li.update)
#proj_node.visualize(
#     source_node=True,
    # rankdir='LR',
    # ratio='.1',
    # labelfontsize='20',
    # nodesep='.1', ranksep='.1',
    # dpi='300'
#)

plt.pause(.1)

center.emit(rot_center)
input()
for pr, th in zip(proj, theta):
    proj_node.emit(pr)
    theta_node.emit(th)
    plt.pause(.01)
print('done')

plt.show()
