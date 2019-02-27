"""Example of using the PCA pipeline on a series of Sine waves"""
import matplotlib.pyplot as plt
from xpdtools.pipelines.qoi import pca_pipeline

from rapidz import Stream

import numpy as np
from matplotlib.lines import Line2D

# Create the streams data goes into
source = Stream()
start = Stream()

# create the pipeline
ns = pca_pipeline(source, start, n_components=.9)

# plot the data
fig, axs = plt.subplots(1, 3)
fig.tight_layout()


def plot_f(data):
    axs[0].cla()
    for i in range(data.shape[0]):
        axs[0].plot(data[i, :] + i)


def plot_g(data):
    axs[1].cla()
    for i, m in zip(range(data.shape[1]), Line2D.filled_markers):
        axs[1].plot(data[:, i], marker=m)


def plot_h(data):
    axs[2].plot(data)


ns["components"].sink(plot_f)
ns["scores"].sink(plot_g)
source.sink(plot_h)
# ns['scores'].sink(axs[1].imshow, aspect='auto')

z = np.linspace(0, np.pi * 2, 100)
xs = [np.sin(z + zz) + np.random.random(100)*.5 for zz in z]

# Run the data into the pipeline
for xx in xs:
    source.emit(xx)
    plt.pause(.1)
#    input()
plt.show()
