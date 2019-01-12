#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from typing import List, NamedTuple, Union, Optional, Tuple
import re
import scipy.ndimage.filters

TrainMeasurement = NamedTuple('TrainMeasurement', [
    ('file_name', str),
    ('label', str),
    ('start_time_label', str),
    ('start_time', float),
    ('epochs', List[int]),
    ('times', List[float]),
    ('losses', List[float]),
])

TestMeasurement = NamedTuple('TestMeasurement', [
    ('label', str),
    ('times', List[float]),
    ('losses', List[float]),
])

_DIRNAME = os.path.abspath(os.path.dirname(__file__))

def plot_measurements(train_measurements, test_measurements, blur):
    # type: (List[TrainMeasurement], List[TestMeasurement], float) -> None

    all_labels = {t.label for t in train_measurements} | {t.label for t in test_measurements}
    label_colors = {l: i for (i, l) in enumerate(all_labels)}

    def plot_measurement(idx, measurement, typ):
        # type: (int, Union[TrainMeasurement, TestMeasurement], str) -> None
        color = 'C{}'.format(idx)
        times = np.array(measurement.times) / 3600
        losses = np.array(measurement.losses)
        if blur > 0 and False:
            losses = scipy.ndimage.filters.gaussian_filter1d(losses, blur)
        label = measurement.label
        plt.plot(times, losses, label='{} {}'.format(typ, label), color=color)

    for m_idx, train_measurement in enumerate(train_measurements):
        plot_measurement(m_idx, train_measurement, 'Train')

        color = 'C{}'.format(label_colors[train_measurement.label])
        times = np.array(train_measurement.times) / 3600
        epochs = np.array(train_measurement.epochs)
        losses = np.array(train_measurement.losses)

        ep_advance = np.where(np.diff(epochs))[0] + 1
        for idx in ep_advance:
            x = times[idx]
            plt.plot([x,x], [losses[idx] - 0.005, losses[idx] + 0.005], color=color)

    for m_idx, test_measurement in enumerate(test_measurements):
        plot_measurement(m_idx + len(train_measurements), test_measurement, 'Test')

    plt.title('Loss over time')
    plt.xlabel('Time in hours')
    plt.ylabel('Loss (MSE / actual)')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.show()

TRAIN_FILE_PAT = re.compile(r'(?P<base>.*?)_(?P<time>\d\d-\d\d-\d\d_\d\d:\d\d:\d\d).log$')

def extract_train_measurement(fname):
    # type: (str) -> Optional[TrainMeasurement]
    with open(fname) as f:
        train_datum = [] # type: List[List[float]]
        for line in f.readlines():
            train_datum.append(list(map(float, line.split())))

        if not train_datum:
            return None

        epochs, times, losses = list(zip(*train_datum))[:3]
        match = TRAIN_FILE_PAT.search(os.path.basename(fname))
        if not match:
            return None

        time_label = match.group('time')
        start_time = time.mktime(time.strptime(time_label, '%m-%d-%y_%H:%M:%S'))
        label = match.group('base')
        return TrainMeasurement(
            os.path.abspath(fname),
            label,
            time_label,
            start_time,
            epochs,
            times,
            losses,
        )

def get_measurements(files):
    # type: (List[str]) -> Tuple[List[TrainMeasurement], List[TestMeasurement]]
    train_measurements = {}
    test_measurements = {}

    for fname in files:
        train_measurement = extract_train_measurement(fname)
        if (train_measurement and (
                train_measurement.label not in train_measurements or
                train_measurements[train_measurement.label] < train_measurement.start_time
        )):
            train_measurements[train_measurement.label] = train_measurement

    for (label, train_measurement) in train_measurements.items():
        checkpoint_path = os.path.join(_DIRNAME, 'test_loss_checkpoint_reports')
        times = []
        losses = []

        from tqdm import tqdm

        for fname in tqdm(os.listdir(checkpoint_path)):
            checkpoint_fname_pat = re.compile(r'{}_{}.mdl_checkpoint_(?P<time>\d+\.\d+).report$'.format(
                re.escape(label),
                re.escape(train_measurement.start_time_label),
            ))
            match = checkpoint_fname_pat.search(os.path.basename(fname))
            if not match:
                continue

            elapsed_time = float(match.group('time'))
            with open(os.path.join(checkpoint_path, fname)) as f:
                line = f.readlines()[-1]
                loss = float(line[1:line.index(']')])
            times.append(elapsed_time)
            losses.append(loss)

        times = np.array(times)
        losses = np.array(losses)
        sorted_idxs = np.argsort(times)
        times = times[sorted_idxs]
        losses = losses[sorted_idxs]
        test_measurements[label] = TestMeasurement(label, times, losses)

    return train_measurements.values(), test_measurements.values()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blur', type=float, default=25)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    train_measurements, test_measurements = get_measurements(args.files)
    plot_measurements(train_measurements, test_measurements, args.blur)

if __name__ == '__main__':
    main()
