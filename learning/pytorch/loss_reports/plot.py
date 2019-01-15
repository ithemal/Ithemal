#!/usr/bin/env python3

from matplotlib import pyplot as plt
from typing import List, NamedTuple, Union, Optional, Tuple
import argparse
import numpy as np
import os
import re
import scipy.ndimage.filters
import subprocess
import time

TrainMeasurement = NamedTuple('TrainMeasurement', [
    ('experiment_name', str),
    ('epochs', List[int]),
    ('times', List[float]),
    ('losses', List[float]),
    ('trainers', List[int]),
])

TestMeasurement = NamedTuple('TestMeasurement', [
    ('experiment_name', str),
    ('times', List[float]),
    ('losses', List[float]),
])

_DIRNAME = os.path.abspath(os.path.dirname(__file__))

_TRAIN = 'Train'
_TEST = 'Test'

def plot_measurements(train_measurements, test_measurements, train_blur, test_blur):
    # type: (List[TrainMeasurement], List[TestMeasurement], float, float) -> None

    def get_times_and_losses(measurement, blur):
        # type: (Union[TrainMeasurement, TestMeasurement], float) -> Tuple[np.array, np.array]
        times = np.array(measurement.times) / 3600
        if blur > 0:
            losses = scipy.ndimage.filters.gaussian_filter1d(measurement.losses, blur)
        else:
            losses = measurement.losses
        return times, losses


    for idx, (train_measurement, test_measurement) in enumerate(zip(train_measurements, test_measurements)):
        train_color = 'C{}'.format(2 * idx)
        test_color = 'C{}'.format(2 * idx + 1)

        train_times, train_losses = get_times_and_losses(train_measurement, train_blur)
        test_times, test_losses = get_times_and_losses(test_measurement, test_blur)
        plt.plot(train_times, train_losses, label='{} train'.format(train_measurement.experiment_name), color=train_color)
        plt.plot(test_times, test_losses, label='{} test'.format(test_measurement.experiment_name), color=test_color)

        ep_advance = np.where(np.diff(train_measurement.epochs))[0] + 1
        for idx in ep_advance:
            x = train_times[idx]
            plt.plot([x,x], [train_losses[idx] - 0.005, train_losses[idx] + 0.005], color=train_color)

    plt.title('Loss over time')
    plt.xlabel('Time in hours')
    plt.ylabel('Loss (MSE / actual)')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.show()

def synchronize_experiment_files(experiment_name):
    # type: (str) -> Tuple[str, List[str]]

    match = re.match(r'^(?P<experiment_name>.*?)(:?\+(?P<time_count>\d+))?$', experiment_name)
    experiment_name = match.group('experiment_name')
    if match.group('time_count'):
        time_count = max(int(match.group('time_count')), 1)
    else:
        time_count = 1

    try:
        output = subprocess.check_output(['aws', 's3', 'ls', 's3://ithemal-experiments/{}/'.format(experiment_name)]).strip()
    except subprocess.CalledProcessError:
        raise ValueError('Unknown experiment {}'.format(experiment_name))

    if isinstance(output, bytes):
        output = output.decode('utf8') # type: ignore

    times = [line.strip().split()[1][:-1] for line in output.split('\n')]
    experiment_times = sorted(times)[-time_count:]

    for experiment_time in experiment_times:
        subprocess.check_call(['aws', 's3', 'sync', 's3://ithemal-experiments/{}/{}'.format(experiment_name, experiment_time),
                               os.path.join(_DIRNAME, 'data', experiment_name, experiment_time),
                               '--exclude', '*', '--include', 'loss_report.log'])

        subprocess.check_call(['aws', 's3', 'sync', 's3://ithemal-experiments/{}/{}/checkpoint_reports'.format(experiment_name, experiment_time),
                               os.path.join(_DIRNAME, 'data', experiment_name, experiment_time, 'checkpoint_reports')])

    return experiment_name, experiment_times

def extract_train_measurement(experiment_name, experiment_time):
    # type: (str, str) -> TrainMeasurement

    fname = os.path.join(_DIRNAME, 'data', experiment_name, experiment_time, 'loss_report.log')

    epochs = []
    times = []
    losses = []
    trainers = []

    with open(fname) as f:
        for line in f.readlines():
            split = line.split()

            epochs.append(int(split[0]))
            times.append(float(split[1]))
            losses.append(float(split[2]))
            trainers.append(int(split[3]))

    return TrainMeasurement(
        experiment_name,
        np.array(epochs),
        np.array(times),
        np.array(losses),
        np.array(trainers),
    )

def extract_test_measurement(experiment_name, experiment_time):
    # type: (str, str) -> TestMeasurement

    checkpoint_fname_pat = re.compile('(?P<time>\d+\.\d+).report')

    times = []
    losses = []
    checkpoint_reports_dir = os.path.join(_DIRNAME, 'data', experiment_name, experiment_time, 'checkpoint_reports')

    for checkpoint_report in os.listdir(checkpoint_reports_dir):
        checkpoint_report = os.path.basename(checkpoint_report)

        match = checkpoint_fname_pat.search(checkpoint_report)

        if not match:
            raise ValueError('Invalid checkpoint report name {} (in {}/{})'.format(checkpoint_report, experiment_name, experiment_time))

        elapsed_time = float(match.group('time'))

        with open(os.path.join(checkpoint_reports_dir, checkpoint_report)) as f:
            line = f.readlines()[-1]
            loss = float(line[1:line.index(']')])
            times.append(elapsed_time)
            losses.append(loss)

    times = np.array(times)
    losses = np.array(losses)
    sorted_idxs = np.argsort(times)
    times = times[sorted_idxs]
    losses = losses[sorted_idxs]

    return TestMeasurement(experiment_name, times, losses)

def get_measurements(experiments):
    # type: (List[str]) -> Tuple[List[TrainMeasurement], List[TestMeasurement]]

    train_measurements = [] # type: List[TrainMeasurement]
    test_measurements = [] # type: List[TestMeasurement]

    for experiment_name in experiments:
        name, experiment_times = synchronize_experiment_files(experiment_name)
        for experiment_time in experiment_times:
            train_measurements.append(extract_train_measurement(name, experiment_time))
            test_measurements.append(extract_test_measurement(name, experiment_time))

    return train_measurements, test_measurements

def main():
    # type: () -> None

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-blur', type=float, default=25)
    parser.add_argument('--test-blur', type=float, default=0.5)
    parser.add_argument('experiments', nargs='+')
    args = parser.parse_args()

    train_measurements, test_measurements = get_measurements(args.experiments)

    plot_measurements(train_measurements, test_measurements, args.train_blur, args.test_blur)

if __name__ == '__main__':
    main()
