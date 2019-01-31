#!/usr/bin/env python

import argparse
import subprocess
import os
import sys
from typing import Any, Dict, List

from aws_utils.instance_utils import format_instance, AwsInstance
import connect_instance

_DIRNAME = os.path.abspath(os.path.dirname(__file__))
_GITROOT = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=_DIRNAME).strip())

class InstanceSynchronizer(connect_instance.InstanceConnectorABC):
    def __init__(self, identity, direction, files):
        # type: (str, str, List[str]) -> None

        super(InstanceSynchronizer, self).__init__(identity, require_pem=True)
        if direction not in ('to', 'from'):
            raise ValueError('Direction "{}" must be either "to" or "from"'.format(direction))
        self.direction = direction

        files = list(map(os.path.abspath, files))
        if not os.path.commonprefix(files + [_GITROOT]) == _GITROOT:
            raise ValueError('All files must be inside of the Ithemal directory!')

        files = list(map(lambda x: os.path.relpath(x, _GITROOT), files))

        self.files = files

    def connect_to_instance(self, instance):
        # type: (Dict[str, Any]) -> None

        ssh_address = 'ec2-user@{}'.format(instance['PublicDnsName'])

        if self.direction == 'to':
            ssh_command = 'cd ithemal; cat | tar --warning=no-unknown-keyword -xz'

            tar = subprocess.Popen(['tar', 'cz'] + self.files, cwd=_GITROOT, stdout=subprocess.PIPE)
            ssh = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, ssh_address, ssh_command], stdin=tar.stdout)
        elif self.direction == 'from':
            ssh_command = 'cd ithemal; tar cz {}'.format(' '.join(map(lambda f: "'{}'".format(f), self.files)))

            ssh = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, ssh_address, ssh_command], stdout=subprocess.PIPE)
            tar = subprocess.Popen(['tar', 'xz'] + self.files, cwd=_GITROOT, stdin=ssh.stdout)

        tar.wait()
        ssh.wait()


def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Synchronize files in the Ithemal directory to a running AWS EC2 instance')

    direction_group = parser.add_mutually_exclusive_group(required=True)
    direction_group.add_argument('--to', help='Send files to the instance', default=False, action='store_true')
    direction_group.add_argument('--from', help='Pull files from the instance', default=False, action='store_true')

    parser.add_argument('identity', help='Identity to use to connect')
    parser.add_argument('--all', help='Synchronize with all instances', default=False, action='store_true')
    parser.add_argument('file', help='Files to synchronize', nargs='+')
    args = parser.parse_args()

    if args.to:
        direction = 'to'
    else:
        direction = 'from'

    synchronizer = InstanceSynchronizer(args.identity, direction, args.file)

    if args.all:
        for instance in synchronizer.get_running_instances():
            synchronizer.connect_to_instance(instance)
    else:
        connect_instance.interactively_connect_to_instance(synchronizer)

if __name__ == '__main__':
    main()
