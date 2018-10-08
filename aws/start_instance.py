#!/usr/bin/env python

import argparse
import json
import os
import subprocess
import sys
import time

from instance_utils import format_instance, AwsInstance

# Ithemal runs on Python 2 mostly
try:
    input = raw_input
except NameError:
    pass

_DIRNAME = os.path.abspath(os.path.dirname(__file__))

class InstanceMaker(AwsInstance):
    def __init__(self, identity, force):
        super(InstanceMaker, self).__init__(identity, require_pem=True)
        self.force = force

    def start_instance(self):
        if not self.force:
            running_instances = self.get_running_instances()
            if running_instances:
                print('You already have {} running instances:'.format(len(running_instances)))
                for instance in running_instances:
                    print(format_instance(instance))
                try:
                    res = input('Would you still like to continue? (y/n) ').lower()[0]
                except KeyboardInterrupt:
                    print('Not creating a new instance')
                    return

                if res[0] != 'y':
                    print('Not creating a new instance')
                    return

        args = ['aws', 'ec2', 'run-instances', '--instance-type', 't2.large', '--key-name', self.identity, '--image-id', 'ami-0b59bfac6be064b78', '--tag-specifications', 'ResourceType="instance",Tags=[{Key="Name",Value="Ithemal Container"}]', '--security-group-ids', 'sg-0780fe1760c00d96d']
        output = subprocess.check_output(args).encode('utf-8')
        parsed_output = json.loads(output)
        instance = parsed_output['Instances'][0]
        instance_id = instance['InstanceId']

        subprocess.check_call(['aws', 'ec2', 'wait', 'instance-running', '--instance-ids', instance_id])

        instance = next(instance for instance in self.get_running_instances() if instance['InstanceId'] == instance_id)

        # wait for SSH to actually become available
        while subprocess.call(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, 'ec2-user@{}'.format(instance['PublicDnsName']), 'exit'],
                              stdout=open(os.devnull, 'w'),
                              stderr=open(os.devnull, 'w'),
        ):
            time.sleep(5)

        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=_DIRNAME).encode('utf-8').strip()
        ls_files = subprocess.Popen(['git', 'ls-files'], cwd=git_root, stdout=subprocess.PIPE)
        tar = subprocess.Popen(['tar', 'Tcz', '-'], cwd=git_root, stdin=ls_files.stdout, stdout=subprocess.PIPE)
        ssh = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, 'ec2-user@{}'.format(instance['PublicDnsName']),
                                'mkdir ithemal; cd ithemal; cat | tar xz; aws/remote_setup.sh'], stdin=tar.stdout)
        ls_files.wait()
        tar.wait()
        ssh.wait()


def main():
    parser = argparse.ArgumentParser(description='Create an AWS instance to run Ithemal')
    parser.add_argument('-f', '--force', help='Make a new instance without worrying about old instances', default=False, action='store_true')
    parser.add_argument('identity', help='Key identity to create with')
    args = parser.parse_args()

    instance_maker = InstanceMaker(args.identity, args.force)
    instance_maker.start_instance()

if __name__ == '__main__':
    main()
