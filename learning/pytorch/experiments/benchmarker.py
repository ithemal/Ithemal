#!/usr/bin/env python

import argparse
import experiment
import subprocess
import os
from typing import List, NamedTuple, Optional

ReportParameters = NamedTuple('ReportParameters', [
    ('remote_model_uri', str),
    ('local_model_uri', str),
    ('remote_report_uri', str),
    ('local_report_uri', str),
])

class Benchmarker(object):
    def __init__(self, name, time, checkpoint=None):
        # type: (str, str, Optional[str]) -> None
        self.name = name
        self.time = time
        self.checkpoint = checkpoint

    def get_checkpoint_report_params(self, expt, iaca_only):
        # type: (experiment.Experiment, bool) -> ReportParameters
        assert self.checkpoint is not None

        checkpoint_dir = expt.checkpoint_file_dir()
        experiment.mkdir(checkpoint_dir)

        s3_checkpoint_path = os.path.join(
            expt.name,
            expt.time,
            'checkpoints',
            '{}.mdl'.format(self.checkpoint)
        )

        remote_model_uri = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, s3_checkpoint_path)
        local_model_uri = os.path.join(checkpoint_dir, '{}.mdl'.format(self.checkpoint))

        if iaca_only:
            report_name = '{}_iaca_only.report'.format(self.checkpoint)
        else:
            report_name = '{}.report'.format(self.checkpoint)

        s3_checkpoint_report_path = os.path.join(expt.name, expt.time, 'checkpoint_reports', report_name)
        remote_report_uri = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, s3_checkpoint_report_path)
        local_report_uri = os.path.join(checkpoint_dir, report_name)

        return ReportParameters(
            remote_model_uri=remote_model_uri,
            local_model_uri=local_model_uri,
            remote_report_uri=remote_report_uri,
            local_report_uri=local_report_uri,
        )

    def get_trained_report_params(self, expt, iaca_only):
        # type: (experiment.Experiment, bool) -> ReportParameters
        expt_root_dir = expt.experiment_root_path()
        experiment.mkdir(expt_root_dir)

        remote_model_uri = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, os.path.join(
            expt.name,
            expt.time,
            'trained.mdl'
        ))
        local_model_uri = os.path.join(expt_root_dir, 'trained.mdl')

        if iaca_only:
            report_name = 'trained_iaca_only.report'
        else:
            report_name = 'trained.report'

        s3_report_path = os.path.join(expt.name, expt.time, report_name)
        remote_report_uri = experiment.get_s3_url(experiment.EXPERIMENT_BUCKET, s3_report_path)
        local_report_uri = os.path.join(expt_root_dir, report_name)

        return ReportParameters(
            remote_model_uri=remote_model_uri,
            local_model_uri=local_model_uri,
            remote_report_uri=remote_report_uri,
            local_report_uri=local_report_uri,
        )

    def benchmark(self, iaca_only):
        # type: (bool) -> None

        expt = experiment.Experiment.make_experiment_from_name_and_time(self.name, self.time)
        expt.download_data()

        if self.checkpoint:
            report_params = self.get_checkpoint_report_params(expt, iaca_only)
        else:
            report_params = self.get_trained_report_params(expt, iaca_only)

        subprocess.check_call(['aws', 's3', 'cp', report_params.remote_model_uri, report_params.local_model_uri])

        validate_args = ['validate', '--load-file', report_params.local_model_uri]
        if iaca_only:
            validate_args.append('--iaca-only')

        with open(report_params.local_report_uri, 'w', 1) as f:
            subprocess.check_call(
                expt.get_ithemal_command_root()
                + expt.base_args
                + validate_args,
                stdout=f
            )


        subprocess.check_call(['aws', 's3', 'cp', report_params.local_report_uri, report_params.remote_report_uri])

def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Get the test performance of a given experiment checkpoint')
    parser.add_argument('name', help='The name of the experiment')
    parser.add_argument('time', help='The time the experiment was run')
    parser.add_argument('--checkpoint', help='The time of the checkpoint. Leave blank to run on trained model')
    parser.add_argument('--iaca-only', help='Whether to test on purely IACA data', action='store_true', default=False)

    args = parser.parse_args()

    benchmarker = Benchmarker(args.name, args.time, args.checkpoint)
    benchmarker.benchmark(args.iaca_only)

if __name__ == '__main__':
    main()
