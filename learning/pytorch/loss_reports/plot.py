#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from typing import List, NamedTuple
import re
import scipy.ndimage.filters

Measurement = NamedTuple('Measurement', [
    ('label', str),
    ('start_time', float),
    ('epochs', List[int]),
    ('times', List[float]),
    ('losses', List[float]),
])

def plot_measurements(measurements):
    for midx, measurement in enumerate(measurements):
        color = 'C{}'.format(midx)
        epochs = np.array(measurement.epochs)
        times = np.array(measurement.times)
        losses = np.array(measurement.losses)
        losses = scipy.ndimage.filters.gaussian_filter1d(losses, 25)
        label = measurement.label
        plt.plot(times / 60, losses, label=label, color=color)

        ep_advance = np.where(np.diff(epochs))[0] + 1
        for idx in ep_advance:
            x = times[idx] / 60
            plt.plot([x,x], [losses[idx] - 0.005, losses[idx] + 0.005], color=color)

    plt.title('Training loss over time')
    plt.xlabel('Time in minutes')
    plt.ylabel('Training loss (MSE / actual)')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.show()

def plot_files(files):
    pat = re.compile(r'(?P<base>.*?)_(?P<time>\d\d-\d\d-\d\d_\d\d:\d\d:\d\d).log$')
    measurements = {}
    for fname in files:
        with open(fname) as f:
            datum = []
            for line in f.readlines():
                datum.append(list(map(float, line.split())))

            if not datum:
                continue

            epochs, times, losses = zip(*datum)
            match = pat.search(os.path.basename(fname))
            if not match:
                continue
            start_time = time.strptime(match.group('time'), '%m-%d-%y_%H:%M:%S')
            label = match.group('base')
            measurements[label] = Measurement(label, start_time, epochs, times, losses)

    plot_measurements(measurements.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    plot_files(args.files)

if __name__ == '__main__':
    main()
