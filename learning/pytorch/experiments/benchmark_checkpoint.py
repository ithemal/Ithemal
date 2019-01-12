#!/usr/bin/env python

import argparse
import experiment
import subprocess
import os
from typing import List

def benchmark_checkpoint(name, time, data, checkpoint, rest_params):
    # type: (str, str, str, str, List[str]) -> None
    expt = experiment.Experiment(name, time, data, [])
    expt.download_data()

    checkpoint_dir = expt.checkpoint_file_dir()

    try:
        os.makedirs(checkpoint_dir)
    except OSError:
        pass

    s3_checkpoint_path = os.path.join(
        expt.name,
        expt.time,
        'checkpoints',
        '{}.mdl'.format(checkpoint)
    )

    s3_checkpoint_url = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, s3_checkpoint_path)
    local_checkpoint_url = os.path.join(checkpoint_dir, '{}.mdl'.format(checkpoint))

    subprocess.check_call(['aws', 's3', 'cp', s3_checkpoint_url, local_checkpoint_url])

    local_checkpoint_report_url = os.path.join(checkpoint_dir, '{}.report'.format(checkpoint))
    with open(local_checkpoint_report_url, 'w') as f:
        subprocess.check_call(
            expt.get_ithemal_command_root()
            + ['validate']
            + rest_params
            + ['--load-file', local_checkpoint_url],
            stdout=f
        )


    s3_checkpoint_report_path = os.path.join(expt.name, expt.time, 'checkpoint_reports')
    s3_checkpoint_report_url = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, s3_checkpoint_report_path)

    subprocess.check_call(['aws', 's3', 'cp', local_checkpoint_report_url, s3_checkpoint_report_url])


def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Get the test performance of a given experiment checkpoint', prefix_chars='?')
    parser.add_argument('name', help='The name of the experiment')
    parser.add_argument('time', help='The time the experiment was run')
    parser.add_argument('data', help='The data to validate against')
    parser.add_argument('checkpoint', help='The time of the checkpoint')
    parser.add_argument('rest', nargs='*', help='Remaining arguments passed to Ithemal')

    args = parser.parse_args()

    benchmark_checkpoint(args.name, args.time, args.data, args.checkpoint, args.rest)

if __name__ == '__main__':
    main()
