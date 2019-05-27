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
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

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

def plot_measurements(train_measurements, test_measurements, has_finished, train_blur, test_blur, plot_trainers, raw_x, save, norm_epoch, min_y, max_y, validation):
    # type: (List[TrainMeasurement], List[TestMeasurement], List[bool], float, float, bool, bool, Optional[str], bool) -> None

    def get_times_and_losses(measurement, blur):
        # type: (Union[TrainMeasurement, TestMeasurement], float) -> Tuple[np.array, np.array]
        times = np.array(measurement.times) / 3600
        if blur > 0:
            losses = scipy.ndimage.filters.gaussian_filter1d(measurement.losses, blur)
        else:
            losses = measurement.losses
        if raw_x:
            return np.arange(len(losses)), losses
        else:
            return times, losses

    plt.title('Loss over time')
    fig = plt.figure(1, figsize=(12.8, 9.6), dpi=100)
    loss_ax = fig.gca()
    if plot_trainers:
        trainer_ax = loss_ax.twinx()
        trainer_ax.set_ylim([1, 6])
        trainer_ax.set_ylabel('Number of running trainers')
    else:
        trainer_ax = None


    if norm_epoch:
        loss_ax.set_xlabel('Epochs')
    else:
        loss_ax.set_xlabel('Time in hours')

    loss_ax.set_ylim([min_y, max_y])
    loss_ax.set_ylabel('Loss')

    for idx, (train_measurement, test_measurement, finished) in enumerate(zip(train_measurements, test_measurements, has_finished)):
        color = 'C{}'.format(idx)
        name = test_measurement.experiment_name
        train_times, train_losses = get_times_and_losses(train_measurement, train_blur)
        test_times, test_losses = get_times_and_losses(test_measurement, test_blur)

        ep_advance = np.where(np.diff(train_measurement.epochs))[0] + 1

        new_test_times = np.empty_like(test_times)

        max_tr = train_times.max()

        if norm_epoch:
            prev = 0
            prev_x = 0
            for k, idx in enumerate(ep_advance):
                x = train_times[idx]
                idxs = (test_times >= prev_x) & (test_times < x)
                old_tests = test_times[idxs]
                new_test_times[idxs] = (old_tests - prev_x) / (x - prev_x) + k
                train_times[prev:idx] = np.linspace(k, k+1, idx - prev)
                prev = idx
                prev_x = x

            idxs = (test_times >= prev_x)
            old_tests = test_times[idxs]
            new_test_times[idxs] = (old_tests - prev_x) / (max_tr - prev_x) + len(ep_advance)
            train_times[prev:] = np.linspace(len(ep_advance), len(ep_advance)+1, len(train_times) - prev)
            test_times = new_test_times
        else:
            for idx in ep_advance:
                x = train_times[idx]
                y = train_losses[idx]
                loss_ax.plot([x,x], [y - 0.005, y + 0.005], color=color)

        loss_ax.plot(train_times, train_losses, label='{} train loss'.format(name), color=color)
        if len(test_times) > 0:
            loss_ax.plot(test_times, test_losses, linestyle='--', label='{} {} loss'.format(name, 'validation' if validation else 'test'), color=color)

        if finished: # or True:
            loss_ax.scatter(train_times[-1:], train_losses[-1:], marker='x', color=color)

        if trainer_ax is not None:
            trainer_ax.plot(train_times, train_measurement.trainers, label='{} trainers'.format(name), color=color)


    loss_ax.legend()

    if save:
        plt.savefig(save)
    else:
        plt.show()

def synchronize_experiment_files(experiment_name):
    # type: (str) -> Tuple[str, List[str], List[bool]]

    match = re.match(r'^(?P<experiment_name>.*?)(:?\+(?P<time_count>\d+))?$', experiment_name)
    if match is None:
        raise ValueError('Unrecognized format: {}'.format(experiment_name))

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

    splits = [line.strip().split() for line in output.split('\n')]
    times = [split[1][:-1] for split in splits if split[0] == 'PRE']

    experiment_times = sorted(times)[-time_count:]
    has_finished = [] # type: List[bool]

    for experiment_time in experiment_times:
        subprocess.check_call(['aws', 's3', 'sync', 's3://ithemal-experiments/{}/{}'.format(experiment_name, experiment_time),
                               os.path.join(_DIRNAME, 'data', experiment_name, experiment_time),
                               '--exclude', '*', '--include', 'loss_report.log'])

        subprocess.check_call(['aws', 's3', 'sync', 's3://ithemal-experiments/{}/{}/checkpoint_reports'.format(experiment_name, experiment_time),
                               os.path.join(_DIRNAME, 'data', experiment_name, experiment_time, 'checkpoint_reports')])

        has_validation_results_code = subprocess.call(
            ['aws', 's3', 'ls', 's3://ithemal-experiments/{}/{}/validation_results.txt'.format(experiment_name, experiment_time)],
            stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'),
        )

        has_finished.append(has_validation_results_code == 0)

    return experiment_name, experiment_times, has_finished

def extract_train_measurement(experiment_name, user_provided_name, experiment_time):
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
        user_provided_name,
        np.array(epochs),
        np.array(times),
        np.array(losses),
        np.array(trainers),
    )

def extract_test_measurement(experiment_name, user_provided_name, experiment_time):
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

    return TestMeasurement(user_provided_name, times, losses)

def get_measurements(experiments, names):
    # type: (List[str], List[str]) -> Tuple[List[TrainMeasurement], List[TestMeasurement], List[bool]]

    train_measurements = [] # type: List[TrainMeasurement]
    test_measurements = [] # type: List[TestMeasurement]
    has_finished = [] # type: List[bool]

    if not names:
        names = experiments

    assert len(names) == len(experiments)

    for experiment_name, user_name in zip(experiments, names):
        name, experiment_times, finished = synchronize_experiment_files(experiment_name)
        has_finished.extend(finished)
        for experiment_time in experiment_times:
            train_measurements.append(extract_train_measurement(name, user_name, experiment_time))
            test_measurements.append(extract_test_measurement(name, user_name, experiment_time))

    return train_measurements, test_measurements, has_finished

def main():
    # type: () -> None

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-blur', type=float, default=25)
    parser.add_argument('--test-blur', type=float, default=0.5)
    parser.add_argument('--min-y', type=float, default=0.0)
    parser.add_argument('--max-y', type=float, default=0.4)
    parser.add_argument('experiments', nargs='+')
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--trainers', default=False, action='store_true')
    parser.add_argument('--no-test', default=False, action='store_true')
    parser.add_argument('--raw-x', default=False, action='store_true')
    parser.add_argument('--sort', default=False, action='store_true')
    parser.add_argument('--validation', default=False, action='store_true')
    parser.add_argument('--norm-epoch', default=False, action='store_true')
    parser.add_argument('--shortest-trainer', default=False, action='store_true')
    parser.add_argument('--save')

    args = parser.parse_args()

    train_measurements, test_measurements, has_finished = get_measurements(args.experiments, args.names)

    if args.no_test:
        test_measurements = list(TestMeasurement(m.experiment_name, [], []) for m in test_measurements)

    if args.sort:
        idxs = np.argsort([-np.mean(m.losses[len(m.losses)//2:]) for m in train_measurements])
        train_measurements = [train_measurements[i] for i in idxs]
        test_measurements = [test_measurements[i] for i in idxs]
        has_finished = [has_finished[i] for i in idxs]

    if args.shortest_trainer:
        shortest_epoch = min(measurement.epochs[-1] for measurement in train_measurements)
        for tridx, (tr, te) in enumerate(zip(train_measurements, test_measurements)):
            try:
                cut_idx = next(i for (i, e) in enumerate(tr.epochs) if e > shortest_epoch)
            except StopIteration:
                continue

            train_measurements[tridx] = TrainMeasurement(
                tr.experiment_name,
                tr.epochs[:cut_idx],
                tr.times[:cut_idx],
                tr.losses[:cut_idx],
                tr.trainers[:cut_idx],
            )

            cut_time = train_measurements[tridx].times[-1]

            try:
                cut_idx = next(i for (i, t) in enumerate(te.times) if t > cut_time)
            except StopIteration:
                continue

            test_measurements[tridx] = TestMeasurement(
                te.experiment_name,
                te.times[:cut_idx],
                te.losses[:cut_idx],
            )

    plot_measurements(train_measurements, test_measurements, has_finished, args.train_blur, args.test_blur, args.trainers, args.raw_x, args.save, args.norm_epoch, args.min_y, args.max_y, args.validation)

if __name__ == '__main__':
    main()
