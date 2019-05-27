#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
import argparse
import subprocess
import os
import sys
from typing import Any, Dict, List, Union

from aws_utils.instance_utils import format_instance, AwsInstance

# Ithemal runs on Python 2 mostly
try:
    input = raw_input
except NameError:
    pass

class InstanceConnectorABC(AwsInstance):
    __metaclass__ = ABCMeta

    @abstractmethod
    def connect_to_instance(self, instance):
        # type: (Dict[str, Any]) -> None
        return NotImplemented

class InstanceConnector(InstanceConnectorABC):
    def __init__(self, identity, host, root, com):
        # type: (str, str, bool, List[str]) -> None
        super(InstanceConnector, self).__init__(identity, require_pem=True)
        self.host = host
        self.root = root
        self.com = com

    def connect_to_instance(self, instance):
        # type: (Dict[str, Any]) -> None
        ssh_address = 'ec2-user@{}'.format(instance['PublicDnsName'])
        ssh_args = ['ssh', '-X', '-i', self.pem_key, '-t', ssh_address]

        if self.com:
            conn_com = "bash -lc '{}'".format(' '.join(self.com).replace("'", r"\'"))
        else:
            conn_com = "bash -lc '~/ithemal/aws/aws_utils/tmux_attach.sh || /home/ithemal/ithemal/aws/aws_utils/tmux_attach.sh'"

        if self.host:
            ssh_args.append(conn_com)
        else:
            if self.root:
                user = 'root'
            else:
                user = 'ithemal'
            ssh_args.append('sudo docker exec -u {} -it ithemal {}'.format(user, conn_com))

        os.execvp('ssh', ssh_args)
        sys.exit(1)

def list_instances(instances):
    # type: (List[Dict[str, Any]]) -> None
    if not instances:
        print('No instances running!')
        return

    for i, instance in enumerate(instances):
        print('{}) {}'.format(i + 1, format_instance(instance)))


def interactively_connect_to_instance(aws_instances):
    # type: (InstanceConnectorABC) -> None
    while True:
        instances = aws_instances.get_running_instances()
        if not instances:
            print('No instances to connect to!')
            return
        elif len(instances) == 1:
            aws_instances.connect_to_instance(instances[0])
            return

        list_instances(instances)

        try:
            res = input('Enter a number to connect to that instance, or "q" to exit: ')
        except KeyboardInterrupt:
            return
        except EOFError:
            return

        if res[0].lower() == 'q':
            return
        else:
            try:
                index_to_connect = int(res)
            except ValueError:
                print('"{}" is not an integer.'.format(res))
                continue

            if index_to_connect < 1 or index_to_connect > len(instances):
                print('{} is not between 1 and {}.'.format(index_to_connect, len(instances)))
                continue

            instance = instances[index_to_connect - 1]
            aws_instances.connect_to_instance(instance)

            return

def connect_to_instance_id_or_index(aws_instances, id_or_index):
    # type: (InstanceConnectorABC, str) -> None
    instances = aws_instances.get_running_instances()

    if len(instances) == 0:
        print('No instances to connect to!')

    try:
        idx = int(id_or_index)
        if idx <= 0 or idx > len(instances):
            print('Provided index must be in the range [{}, {}]'.format(1, len(instances)))
            return

        aws_instances.connect_to_instance(instances[idx - 1])
    except ValueError:
        pass

    possible_instances = [instance for instance in instances if instance['InstanceId'].startswith(id_or_index)]
    if len(possible_instances) == 0:
        raise ValueError('{} is not a valid instance ID or index'.format(id_or_index))
    elif len(possible_instances) == 1:
        aws_instances.connect_to_instance(possible_instances[0])
    else:
        raise ValueError('Multiple instances have ambiguous identifier prefix {}'.format(id_or_index))

def main():
    # type: () -> None

    parser = argparse.ArgumentParser(description='Connect to a running AWS EC2 instance')

    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument('--host', help='Connect directly to the host', default=False, action='store_true')
    user_group.add_argument('--root', help='Connect to root in the Docker instance', default=False, action='store_true')
    user_group.add_argument('--list', help='Just list the instances, rather than connecting', default=False, action='store_true')

    parser.add_argument('identity', help='Identity to use to connect')
    parser.add_argument('instance_id', help='Instance IDs to manually connect to', nargs='?', default=None)
    parser.add_argument('--com', help='Command to run (uninteractive)', nargs='+')
    args = parser.parse_args()

    aws_instances = InstanceConnector(args.identity, args.host, args.root, args.com)

    if args.list:
        list_instances(aws_instances.get_running_instances())
        return

    if args.instance_id:
        connect_to_instance_id_or_index(aws_instances, args.instance_id)
    else:
        interactively_connect_to_instance(aws_instances)

if __name__ == '__main__':
    main()
