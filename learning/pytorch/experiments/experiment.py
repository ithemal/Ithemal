#!/usr/bin/env python

import argparse
import datetime
import json
import os
import subprocess
import sys
import urlparse
import tempfile
import time
import traceback
from typing import Any, Dict, List, NamedTuple, Optional

_DIRNAME = os.path.abspath(os.path.dirname(__file__))
PYTHON = sys.executable

try:
    ITHEMAL_HOME = os.environ['ITHEMAL_HOME']
except:
    # as a backup (e.g. on Alex's computer) set ITHEMAL_HOME as a function of the gitroot
    ITHEMAL_HOME = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=_DIRNAME).strip()

PYTORCH_HOME = os.path.join(ITHEMAL_HOME, 'learning', 'pytorch')

EXPERIMENT_BUCKET = 'ithemal-experiments'
DATASET_BUCKET = 'ithemal-datasets'
CHECKPOINT_QUEUE = 'checkpoint_queue'

DEBUG = False

_BENCHMARK_CHECKPOINT = os.path.join(
    '${ITHEMAL_HOME}', # whatever ITHEMAL_HOME is on remote machine
    'learning', 'pytorch', 'experiments', 'benchmarker.py'
)

def debug_print(params):
    # type: (List[str]) -> None
    if DEBUG:
        print(' '.join(params))

def get_s3_url(bucket, path):
    # type: (str, str) -> str
    return urlparse.urlunsplit(['s3', bucket, path, '', ''])

def mkdir(directory):
    # type: (str) -> None
    try:
        os.makedirs(directory)
    except OSError:
        pass

class Experiment(object):
    def __init__(self, name, time, data, base_args=[], train_args=[]):
        # type: (str, str, str, List[str], List[str]) -> None
        self.name = name
        self.time = time
        self.data = os.path.basename(data)
        self.base_args = list(map(str, base_args))
        self.train_args = list(map(str, train_args))
        self.proc = None # type: Optional[subprocess.Popen]

    @staticmethod
    def make_experiment_from_name_and_time(experiment_name, experiment_time):
        # type: (str, str) -> Experiment
        remote_config_file_url = get_s3_url(EXPERIMENT_BUCKET, os.path.join(experiment_name, experiment_time, 'config.json'))
        local_config_file_url = os.path.join(PYTORCH_HOME, 'saved', experiment_name, experiment_time, 'config.json')
        subprocess.check_call(['aws', 's3', 'cp', remote_config_file_url, local_config_file_url])
        return Experiment.make_experiment_from_config_file(local_config_file_url, experiment_time=experiment_time)

    @staticmethod
    def make_experiment_from_name(experiment_name):
        # type: (str) -> Experiment
        experiment_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
        remote_config_file_url = get_s3_url(EXPERIMENT_BUCKET, os.path.join(experiment_name, 'config.json'))
        local_config_file_url = os.path.join(PYTORCH_HOME, 'saved', experiment_name, experiment_time, 'config.json')
        subprocess.check_call(['aws', 's3', 'cp', remote_config_file_url, local_config_file_url])
        return Experiment.make_experiment_from_config_file(local_config_file_url, experiment_time=experiment_time)

    @staticmethod
    def make_experiment_from_config_file(config_file, experiment_time=None):
        # type: (str, Optional[str]) -> Experiment

        with open(config_file) as f:
            config = json.load(f)

        if experiment_time is None:
            start_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
        else:
            start_time = experiment_time

        return Experiment(
            config['name'],
            start_time,
            config['dataset'],
            config.get('base_args', []),
            config.get('train_args', []),
        )

    def config_of_experiment(self):
        # type: () -> Dict[str, Any]

        return {
            'name': self.name,
            'dataset': self.data,
            'base_args': self.base_args,
            'train_args': self.train_args,
        }

    def experiment_root_path(self):
        # type: () -> str
        return os.path.join(PYTORCH_HOME, 'saved', self.name, self.time)

    def checkpoint_file_dir(self):
        # type: () -> str
        return os.path.join(self.experiment_root_path(), 'checkpoints')

    def checkpoint_file_name(self, run_time):
        # type: (float) -> str
        return os.path.join(self.checkpoint_file_dir(), '{}.mdl'.format(run_time))

    def s3_root_path(self):
        # type: () -> str
        return get_s3_url(EXPERIMENT_BUCKET, os.path.join(self.name, self.time))

    def get_ithemal_command_root(self):
        # type: () -> List[str]
        return [
            PYTHON, os.path.join(PYTORCH_HOME, 'ithemal', 'run_ithemal.py'),
            '--data', os.path.join(PYTORCH_HOME, 'saved', self.data),
        ]

    def get_params(self):
        # type: () -> List[str]
        return self.get_ithemal_command_root() + self.base_args + [
            'train',
            '--experiment-name', self.name,
            '--experiment-time', self.time,
        ] + self.train_args

    def download_data(self):
        # type: () -> None
        # download the data if not present on this machine
        data_url = get_s3_url(DATASET_BUCKET, '')
        # sync is smarter than cp, but only works on directories: tell it to only sync that one file
        sync_args = ['aws', 's3', 'sync', data_url, os.path.join(PYTORCH_HOME, 'saved'), '--exclude', '*', '--include', self.data]
        debug_print(sync_args)
        subprocess.check_call(sync_args)

    def start_experiment(self):
        # type: () -> None
        self.download_data()
        root = self.experiment_root_path()
        mkdir(root)

        params = self.get_params()

        with open(os.path.join(root, 'config.json'), 'w') as f:
            json.dump(self.config_of_experiment(), f)

        with open(os.path.join(root, 'cmdline'), 'w') as f:
            f.write(' '.join(params))

        debug_print(params)
        # open proc, line buffer stdout

        self.proc = subprocess.Popen(params, stdout=open(os.path.join(root, 'stdout'), 'w', 1))

    def enqueue_checkpoints(self, checkpoint_times):
        # type: (List[str]) -> None

        for checkpoint_time in checkpoint_times:
            command_param = ' '.join([_BENCHMARK_CHECKPOINT, self.name, self.time, '--checkpoint', checkpoint_time])
            params = [
                os.path.join(ITHEMAL_HOME, 'aws', 'command_queue.py'),
                'send', CHECKPOINT_QUEUE, command_param
            ]

            debug_print(params)
            subprocess.call(params, stdout=open('/dev/null', 'w'))

    def sync_all(self):
        # type: () -> None
        params = ['aws', 's3', 'sync', self.experiment_root_path(), self.s3_root_path()]
        debug_print(params)
        subprocess.check_call(params)

    def run_and_sync(self):
        # type: () -> bool

        self.start_experiment()
        proc = self.proc
        if proc is None:
            raise Exception('Process not created!')

        s3_bucket_checkpoint_path = os.path.join(self.s3_root_path(), 'checkpoints')
        checkpoint_path = self.checkpoint_file_dir()
        mkdir(checkpoint_path)

        def sync():
            # type: () -> None

            # sync checkpoints, capturing the new checkpoints and enqueuing them for validation
            params = ['aws', 's3', 'sync', '--no-progress', checkpoint_path, s3_bucket_checkpoint_path]
            debug_print(params)
            checkpoints_output = subprocess.check_output(params).strip()
            if checkpoints_output:
                print('Checkpoints Output: "{}", split: "{}"'.format(checkpoints_output, checkpoints_output.strip().split('\n')))
                checkpoint_files = [line.split()[1] for line in checkpoints_output.split('\n')]
                checkpoint_times = [os.path.basename(fname)[:-len('.mdl')] for fname in checkpoint_files]
                self.enqueue_checkpoints(checkpoint_times)

            self.sync_all()

        while proc.poll() is None:
            sync()
            time.sleep(60)
        sync()

        for iaca_only in (True, False):
            args = [_BENCHMARK_CHECKPOINT, self.name, self.time]
            if iaca_only:
                args.append('--iaca-only')
            params = [
                os.path.join(ITHEMAL_HOME, 'aws', 'command_queue.py'),
                'send', CHECKPOINT_QUEUE, ' '.join(args),
            ]

            debug_print(params)
            subprocess.call(params, stdout=open('/dev/null', 'w'))


        params = [os.path.join(ITHEMAL_HOME, 'aws', 'ping_slack.py'), 'Experiment {}_{} finished with exit code {}'.format(
            self.name,
            self.time,
            proc.returncode,
        )]
        debug_print(params)
        subprocess.check_call(params)

        return proc.returncode == 0

def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Run experiments, syncing with AWS')
    parser.add_argument('experiment', help='Experiment name or file to run')
    args = parser.parse_args()

    if os.path.exists(args.experiment):
        experiment = Experiment.make_experiment_from_config_file(args.experiment)
    else:
        experiment = Experiment.make_experiment_from_name(args.experiment)

    try:
        success = experiment.run_and_sync()
    except:
        success = False
        # catch literally anything (including KeyboardInterrupt, SystemExit)
        traceback.print_exc()

        if experiment.proc is not None:
            try:
                print('Terminating Ithemal process!')
                experiment.proc.terminate()
                experiment.proc.wait()
            except KeyboardInterrupt:
                print('Force killing Ithemal')
                experiment.proc.kill()
    finally:
        print('Synchronizing files...')
        experiment.sync_all()

    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
