#!/usr/bin/env python

import argparse
import subprocess
import os
import sys

from instance_utils import format_instance, AwsInstance

# Ithemal runs on Python 2 mostly
try:
    input = raw_input
except NameError:
    pass

class InstanceConnector(AwsInstance):
    def __init__(self, identity, host, root):
        super(InstanceConnector, self).__init__(identity, require_pem=True)
        self.host = host
        self.root = root

    def connect_to_instance(self, instance):
        ssh_address = 'ec2-user@{}'.format(instance['PublicDnsName'])
        ssh_args = ['ssh', '-i', self.pem_key, '-t', ssh_address]

        conn_com = "bash -lc '~/ithemal/aws/tmux_attach.sh'"

        if self.host:
            ssh_args.append(conn_com)
        elif self.root:
            ssh_args.append('sudo docker exec -u root -it ithemal {}'.format(conn_com))
        else:
            ssh_args.append('sudo docker exec -u ithemal -it ithemal {}'.format(conn_com))

        os.execvp('ssh', ssh_args)
        sys.exit(1)

def interactively_connect_to_instance(aws_instances):
    while True:
        instances = aws_instances.get_running_instances()
        if not instances:
            print('No instances to connect to!')
            return
        elif len(instances) == 1:
            aws_instances.connect_to_instance(instances[0])
            return

        print('Active instances:')
        for i, instance in enumerate(instances):
            print('{}) {}'.format(i + 1, format_instance(instance)))

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
                print('{} is not between 1 and {}.'.format(index_to_connect, len(instances) + 1))
                continue

            instance = instances[index_to_connect - 1]
            aws_instances.connect_to_instance(instance)

            return

def main():
    parser = argparse.ArgumentParser(description='Connect to a running AWS EC2 instance')

    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument('--host', help='Connect directly to the host', default=False, action='store_true')
    user_group.add_argument('--root', help='Connect to root in the Docker instance', default=False, action='store_true')

    parser.add_argument('identity', help='Identity to use to connect')
    parser.add_argument('instance_id', help='Instance IDs to manually connect to', nargs='?', default=None)
    args = parser.parse_args()

    aws_instances = InstanceConnector(args.identity, args.host, args.root)

    if args.instance_id:
        instance = next(instance for instance in aws_instances.get_running_instances() if instance['InstanceId'] == args.instance_id)
        aws_instances.connect_to_instance(instance)
    else:
        interactively_connect_to_instance(aws_instances)

if __name__ == '__main__':
    main()