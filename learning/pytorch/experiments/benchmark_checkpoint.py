#!/usr/bin/env python

import argparse
import experiment
import subprocess
import os
from typing import List

def benchmark_checkpoint(name, time, checkpoint):
    # type: (str, str, str) -> None
    expt = experiment.Experiment.make_experiment_from_name_and_time(name, time)
    expt.download_data()

    checkpoint_dir = expt.checkpoint_file_dir()
    experiment.mkdir(checkpoint_dir)

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
            + expt.base_args
            + ['validate' '--load-file', local_checkpoint_url],
            stdout=f
        )


    s3_checkpoint_report_path = os.path.join(expt.name, expt.time, 'checkpoint_reports', '{}.report'.format(checkpoint))
    s3_checkpoint_report_url = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, s3_checkpoint_report_path)

    subprocess.check_call(['aws', 's3', 'cp', local_checkpoint_report_url, s3_checkpoint_report_url])


def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Get the test performance of a given experiment checkpoint')
    parser.add_argument('name', help='The name of the experiment')
    parser.add_argument('time', help='The time the experiment was run')
    parser.add_argument('checkpoint', help='The time of the checkpoint')

    args = parser.parse_args()

    benchmark_checkpoint(args.name, args.time, args.checkpoint)

if __name__ == '__main__':
    main()
