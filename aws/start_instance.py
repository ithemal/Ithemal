#!/usr/bin/env python

from __future__ import print_function

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional
from aws_utils.instance_utils import format_instance, AwsInstance
import command_queue

# Ithemal runs on Python 2 mostly
try:
    input = raw_input
except NameError:
    pass

_DIRNAME = os.path.abspath(os.path.dirname(__file__))

class InstanceMaker(AwsInstance):
    def __init__(self, identity, name, instance_type, db, force, no_connect, spot, queue_name):
        # type: (str, str, str, str, bool, bool, Optional[int], str) -> None
        super(InstanceMaker, self).__init__(identity, require_pem=True)
        self.name = name
        self.instance_type = instance_type
        self.db = db
        self.force = force
        self.no_connect = no_connect
        self.spot = spot
        self.queue_name = queue_name

    def start_instance(self):
        # type: () -> None

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

        name = 'Ithemal'
        if self.name:
            name += ': {}'.format(self.name)

        block_device_mappings = [{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 16}}]
        iam_profile_name = 'ithemal-ec2'
        iam_profile_struct = {'Name': iam_profile_name}

        if self.spot:
            launch_specification = {
                'InstanceType': self.instance_type,
                'SecurityGroupIds': ['sg-0780fe1760c00d96d'],
                'BlockDeviceMappings': block_device_mappings,
                'KeyName': self.identity,
                'ImageId': 'ami-0b59bfac6be064b78',
                'IamInstanceProfile': iam_profile_struct,
            }
            run_com = lambda com: json.loads(subprocess.check_output(com))['SpotInstanceRequests'][0]
            com = [
                'aws', 'ec2', 'request-spot-instances',
                '--launch-specification', json.dumps(launch_specification)
            ]
            if self.spot > 0:
                com.extend(['--block-duration-minutes', str(self.spot * 60)])
            output = run_com(com)
            print('Submitted spot instance request')

            try:
                while 'InstanceId' not in output:
                    print('\rWaiting for spot request to be fulfilled ({})...'.format(
                        output['Status']['Code']
                    ), end=' ' * 20 + '\r')

                    time.sleep(1)
                    output = run_com([
                        'aws', 'ec2', 'describe-spot-instance-requests',
                        '--spot-instance-request-ids', output['SpotInstanceRequestId'],
                    ])

            except (KeyboardInterrupt, SystemExit):
                subprocess.check_call([
                    'aws', 'ec2', 'cancel-spot-instance-requests',
                    '--spot-instance-request-ids', output['SpotInstanceRequestId'],
                ])
                sys.exit(1)

            print() # clear status message

            instance_id = output['InstanceId']
            # set the name, since spot instances don't let us do that in the creation request
            subprocess.check_call([
                'aws', 'ec2', 'create-tags',
                '--resources', instance_id,
                '--tags', 'Key=Name,Value="{}"'.format(name)
            ])
        else:
            args = [
                'aws', 'ec2', 'run-instances',
                '--instance-type', self.instance_type,
                '--key-name', self.identity,
                '--image-id', 'ami-0b59bfac6be064b78',
                '--tag-specifications', 'ResourceType="instance",Tags=[{{Key="Name",Value="{}"}}]'.format(name),
                '--security-group-ids', 'sg-0780fe1760c00d96d',
                '--block-device-mappings', json.dumps(block_device_mappings),
                '--iam-instance-profile', json.dumps(iam_profile_struct),
            ]
            output = subprocess.check_output(args)
            parsed_output = json.loads(output)
            instance = parsed_output['Instances'][0]
            instance_id = instance['InstanceId']

        print('Started instance! Waiting for connection...')

        subprocess.check_call(['aws', 'ec2', 'wait', 'instance-running', '--instance-ids', instance_id])

        instance = next(instance for instance in self.get_running_instances() if instance['InstanceId'] == instance_id)
        ssh_address = 'ec2-user@{}'.format(instance['PublicDnsName'])

        # wait for SSH to actually become available
        while subprocess.call(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, ssh_address, 'exit'],
                              stdout=open(os.devnull, 'w'),
                              stderr=open(os.devnull, 'w'),
        ):
            time.sleep(1)


        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=_DIRNAME).strip()
        ls_files = subprocess.Popen(['git', 'ls-files'], cwd=git_root, stdout=subprocess.PIPE)
        tar = subprocess.Popen(['tar', 'Tcz', '-'], cwd=git_root, stdin=ls_files.stdout, stdout=subprocess.PIPE)

        aws_credentials = json.loads(subprocess.check_output(['aws', 'ecr', 'get-authorization-token']).strip())
        authorization_datum = aws_credentials['authorizationData'][0]
        aws_authorization = base64.b64decode(authorization_datum['authorizationToken'])
        aws_authorization_user = aws_authorization[:aws_authorization.index(':')]
        aws_authorization_token = aws_authorization[aws_authorization.index(':')+1:]
        aws_endpoint = authorization_datum['proxyEndpoint']

        region = subprocess.check_output(['aws', 'configure', 'get', 'region']).strip()

        mysql_credentials_dict = json.loads(subprocess.check_output(['aws', 'secretsmanager', 'get-secret-value', '--secret-id', 'ithemal/mysql-{}'.format(self.db)]).strip())
        mysql_credentials = json.loads(mysql_credentials_dict['SecretString'])
        mysql_user = mysql_credentials['username']
        mysql_password = mysql_credentials['password']
        mysql_host = mysql_credentials['host']
        mysql_port = mysql_credentials['port']

        initialization_command = 'mkdir ithemal; cd ithemal; cat | tar xz; aws/aws_utils/remote_setup.sh {}'.format(' '.join(map(str, [
            aws_authorization_user,
            aws_authorization_token,
            aws_endpoint,
            mysql_user,
            mysql_password,
            mysql_host,
            mysql_port,
            region,
        ])))

        ssh = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, ssh_address, initialization_command],
                               stdin=tar.stdout)
        ls_files.wait()
        tar.wait()
        ssh.wait()

        if self.queue_name:
            self.start_queue_on_instance(instance, ssh_address)

        if not self.no_connect:
            os.execlp(sys.executable, sys.executable, os.path.join(_DIRNAME, 'connect_instance.py'), self.identity, instance['InstanceId'])

    def start_queue_on_instance(self, instance, ssh_address):
        # type: (Dict[str, Any], str) -> None

        subprocess.check_call([
            'aws', 'ec2', 'create-tags',
            '--resources', instance['InstanceId'],
            '--tags', 'Key=QueueName,Value="{}"'.format(self.queue_name)
        ])

        queue_url = command_queue.queue_url_of_name(self.queue_name)

        subprocess.check_call([
            'ssh', '-i', self.pem_key, ssh_address,
            'sudo docker exec -u ithemal -dit ithemal bash -lc "~/ithemal/aws/aws_utils/queue_process.py --kill {}"'.format(queue_url)
        ])

def main():
    # type: () -> None

    parser = argparse.ArgumentParser(description='Create an AWS instance to run Ithemal')
    parser.add_argument('identity', help='Key identity to create with')
    parser.add_argument('-n', '--name', help='Name to start the container with', default=None)
    parser.add_argument('-t', '--type', help='Instance type to start (default: t2.large)', default='t2.large')
    parser.add_argument('-f', '--force', help='Make a new instance without worrying about old instances', default=False, action='store_true')
    parser.add_argument('-nc', '--no-connect', help='Don\'t connect to the instance after it is started', default=False, action='store_true')
    parser.add_argument('-q', '--queue', metavar='QUEUE_NAME', help='Perform actions consumed from given queue')

    spot_group = parser.add_mutually_exclusive_group()
    spot_group.add_argument('--spot-reserved', '-sr', help='Start a spot instance, reserved for a specific duration (between 1 and 6 hours)', type=int, dest='spot', metavar='DURATION')
    spot_group.add_argument('--spot-preempt', '-sp', help='Start a spot instance, preemptable', action='store_const', const=-1, dest='spot')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prod-ro-db', help='Use the read-only prod database (default)', action='store_true')
    group.add_argument('--prod-db', help='Use the writeable prod database', action='store_true')
    group.add_argument('--dev-db', help='Use the development database', action='store_true')

    args = parser.parse_args()

    if args.prod_db:
        db = 'prod'
    elif args.dev_db:
        db = 'dev'
    else:
        db = 'prod-ro'

    # spot can be either unspecified, -1 (for preemptible), or between 1 and 6
    if args.spot not in (None, -1, 1, 2, 3, 4, 5, 6):
        print('Spot duration must be between 1 and 6 hours')
        return

    instance_maker = InstanceMaker(args.identity, args.name, args.type, db, args.force, args.no_connect, args.spot, args.queue)
    instance_maker.start_instance()

if __name__ == '__main__':
    main()
