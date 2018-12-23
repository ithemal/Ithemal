import json
import os
import subprocess
import time

def format_instance(instance):
    name = 'Unnamed'
    try:
        name_tag = next(tag for tag in instance.get('Tags', []) if tag['Key'] == 'Name')
        name = name_tag['Value']
    except StopIteration:
        pass

    instance_id = instance['InstanceId']
    instance_type = instance['InstanceType']
    launch_time = instance['LaunchTime']
    key_name = instance['KeyName']

    return '{} :: {} :: {} :: {} :: {}'.format(
        name,
        instance_id,
        instance_type,
        launch_time,
        key_name,
    )

class AwsInstance(object):
    def __init__(self, identity, require_pem=False):
        self.identity = identity
        self.pem_key = os.path.expanduser('~/.ssh/{}.pem'.format(identity))
        if require_pem and not os.path.exists(os.path.expanduser(self.pem_key)):
            raise ValueError('Cannot create an AWS instance without the key at {}'.format(self.pem_key))

    def get_running_instances(self):
        args = ['aws', 'ec2', 'describe-instances', '--filters', 'Name=instance-state-name,Values=pending,running']

        if self.identity:
            args.append('Name=key-name,Values={}'.format(self.identity))

        output = subprocess.check_output(args)
        parsed_out = json.loads(output)

        # flatten output
        instances = [instance
                     for reservation in parsed_out['Reservations']
                     for instance in reservation['Instances']]

        def start_time_of_instance(instance):
            return time.strptime(instance['LaunchTime'], '%Y-%m-%dT%H:%M:%S.%fZ')

        return sorted(instances, key=start_time_of_instance)
