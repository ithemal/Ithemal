#!/usr/bin/env python

import argparse
import base64
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
    def __init__(self, identity, name, instance_type, db, force):
        super(InstanceMaker, self).__init__(identity, require_pem=True)
        self.name = name
        self.instance_type = instance_type
        self.db = db
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

        name = 'Ithemal'
        if self.name:
            name += ': {}'.format(self.name)

        args = [
            'aws', 'ec2', 'run-instances',
            '--instance-type', self.instance_type,
            '--key-name', self.identity,
            '--image-id', 'ami-0b59bfac6be064b78',
            '--tag-specifications', 'ResourceType="instance",Tags=[{{Key="Name",Value="{}"}}]'.format(name),
            '--security-group-ids', 'sg-0780fe1760c00d96d',
            '--block-device-mappings', '[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 16}}]',
        ]
        output = subprocess.check_output(args)
        parsed_output = json.loads(output)
        instance = parsed_output['Instances'][0]
        instance_id = instance['InstanceId']

        print('Started instance! Waiting for connection...')

        subprocess.check_call(['aws', 'ec2', 'wait', 'instance-running', '--instance-ids', instance_id])

        instance = next(instance for instance in self.get_running_instances() if instance['InstanceId'] == instance_id)

        # wait for SSH to actually become available
        while subprocess.call(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, 'ec2-user@{}'.format(instance['PublicDnsName']), 'exit'],
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

        mysql_credentials_dict = json.loads(subprocess.check_output(['aws', 'secretsmanager', 'get-secret-value', '--secret-id', 'ithemal/mysql-{}'.format(self.db)]).strip())
        mysql_credentials = json.loads(mysql_credentials_dict['SecretString'])
        mysql_user = mysql_credentials['username']
        mysql_password = mysql_credentials['password']
        mysql_host = mysql_credentials['host']
        mysql_port = mysql_credentials['port']

        initialization_command = 'mkdir ithemal; cd ithemal; cat | tar xz; aws/remote_setup.sh {}'.format(' '.join(map(str, [
            aws_authorization_user,
            aws_authorization_token,
            aws_endpoint,
            mysql_user,
            mysql_password,
            mysql_host,
            mysql_port,
        ])))

        print(initialization_command)
        ssh = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no', '-i', self.pem_key, 'ec2-user@{}'.format(instance['PublicDnsName']), initialization_command],
                               stdin=tar.stdout)
        ls_files.wait()
        tar.wait()
        ssh.wait()

        os.execlp(sys.executable, sys.executable, os.path.join(_DIRNAME, 'connect_instance.py'), self.identity, instance['InstanceId'])


def main():
    parser = argparse.ArgumentParser(description='Create an AWS instance to run Ithemal')
    parser.add_argument('identity', help='Key identity to create with')
    parser.add_argument('-n', '--name', help='Name to start the container with', default=None)
    parser.add_argument('-t', '--type', help='Instance type to start (default: t2.large)', default='t2.large')
    parser.add_argument('-f', '--force', help='Make a new instance without worrying about old instances', default=False, action='store_true')

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

    instance_maker = InstanceMaker(args.identity, args.name, args.type, db, args.force)
    instance_maker.start_instance()

if __name__ == '__main__':
    main()
