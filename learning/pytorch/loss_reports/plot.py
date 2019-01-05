#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
from typing import List, NamedTuple
import re

Measurement = NamedTuple('Measurement', [
    ('trainers', int),
    ('threads', int),
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

        label = '{} Trainers, {} Threads'.format(measurement.trainers, measurement.threads)
        plt.plot(times / 60, losses, label=label, color=color)

        ep_advance = np.where(np.diff(epochs))[0] + 1
        for idx in ep_advance:
            x = times[idx] / 60
            plt.plot([x,x], [losses[idx] - 0.015, losses[idx] + 0.015], color=color)

    plt.title('Training loss over time')
    plt.xlabel('Time in minutes')
    plt.ylabel('Training loss (MSE / actual)')
    plt.legend()
    plt.show()

def plot_files(files):
    pat = re.compile(r'(?P<trainers>\d+)_(?P<threads>\d+)_(?P<time>\d+(?:\.\d+)?)\.log')
    measurements = {}
    for fname in files:
        with open(fname) as f:
            datum = []
            for line in f.readlines():
                datum.append(list(map(float, line.split())))

            if not datum:
                continue

            epochs, times, losses = zip(*datum)
            match = pat.match(os.path.basename(fname))
            trainers = int(match.group('trainers'))
            threads = int(match.group('threads'))
            start_time = float(match.group('time'))

            key = (trainers, threads)
            if key not in measurements or measurements[key].start_time < start_time:
                measurements[key] = Measurement(trainers, threads, start_time, epochs, times, losses)

    plot_measurements(measurements.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    plot_files(args.files)

if __name__ == '__main__':
    main()
