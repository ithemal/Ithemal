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
    def __init__(self, identity):
        super(InstanceConnector, self).__init__(identity, require_pem=True)

    def connect_to_instance(self, instance):
        print('Connecting!')
        print('')

        ssh_address = 'ec2-user@{}'.format(instance['PublicDnsName'])
        os.execlp('ssh', 'ssh', '-i', self.pem_key, ssh_address, 'sudo docker exec -it ithemal bash -l')
        sys.exit(1)

def interactively_connect_to_instance(aws_instances):
    while True:
        instances = aws_instances.get_running_instances()
        if not instances:
            print('No instances to connect to!')
            return
        elif len(instances) == 1:
            aws_instances.connect_to_instance(instances[0])

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

def main():
    parser = argparse.ArgumentParser(description='Kill running AWS EC2 instances')
    parser.add_argument('identity', help='Identity to use to connect')
    parser.add_argument('instance_id', help='Instance IDs to manually connect to', nargs='?', default=None)
    args = parser.parse_args()

    aws_instances = InstanceConnector(args.identity)

    if args.instance_id:
        instance = next(instance for instance in aws_instances.get_running_instances() if instance['InstanceId'] == args.instance_id)
        aws_instances.connect_to_instance(instance)
    else:
        interactively_connect_to_instance(aws_instances)

if __name__ == '__main__':
    main()
