import matplotlib.pyplot as plt
from xpdtools.pipelines.qoi import pca_pipeline

from rapidz import Stream

import numpy as np

# Create the streams data goes into
source = Stream()
start = Stream()

# create the pipeline
ns = pca_pipeline(source, start)

# plot the data
fig, axs = plt.subplots(1, 2)
fig.tight_layout()


def plot_f(data):
    axs[0].cla()
    for i in range(data.shape[0]):
        axs[0].plot(data[i, :])


def plot_g(data):
    print(data.shape)
    axs[1].cla()
    for i, m in zip(range(data.shape[1]), ['.', 'o', '<']):
        axs[1].plot(data[:, i], marker=m)


ns['components'].sink(plot_f)
ns['scores'].sink(plot_g)
# ns['scores'].sink(axs[1].imshow, aspect='auto')

z = np.linspace(0, np.pi * 2, 100)

xs = [np.sin(z + zz) for zz in z]
# Run the data into the pipeline
for x in xs:
    source.emit(x)
    plt.pause(.1)
plt.show()
