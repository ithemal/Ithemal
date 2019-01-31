import calendar
import datetime
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

def utc_to_local_time(dt):
    # type: (datetime.datetime) -> datetime.datetime
    # https://stackoverflow.com/a/13287083
    timestamp = calendar.timegm(dt.timetuple())
    local_dt = datetime.datetime.fromtimestamp(timestamp)
    return local_dt.replace(microsecond=dt.microsecond)

def format_instance(instance):
    # type: (Dict[str, Any]) -> str
    name = 'Unnamed'
    try:
        name_tag = next(tag for tag in instance.get('Tags', {}) if tag['Key'] == 'Name')
        name = name_tag['Value']
    except StopIteration:
        pass

    instance_id = instance['InstanceId']
    instance_type = instance['InstanceType']
    launch_time = instance['LaunchTime']
    key_name = instance['KeyName']
    ip_addr = instance['PublicIpAddress']
    is_spot = bool(instance.get('SpotInstanceRequestId'))

    launch_datetime_utc = datetime.datetime.strptime(launch_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    local_launch_time = utc_to_local_time(launch_datetime_utc).strftime('%m/%d/%Y %H:%M:%S')

    identifiers = [
        name,
        instance_id,
        instance_type,
        local_launch_time,
        ip_addr,
        'Spot' if is_spot else 'On Demand',
    ]

    return ' :: '.join(identifiers)

class AwsInstance(object):
    def __init__(self, identity, require_pem=False):
        # type: (str, bool) -> None

        self.identity = identity
        self.pem_key = os.path.expanduser('~/.ssh/{}.pem'.format(identity))
        if require_pem and not os.path.exists(os.path.expanduser(self.pem_key)):
            raise ValueError('Cannot create an AWS instance without the key at {}'.format(self.pem_key))

    def get_running_instances(self):
        # type: () -> List[Dict[str, Any]]

        args = ['aws', 'ec2', 'describe-instances', '--filters', 'Name=instance-state-name,Values=pending,running']

        if self.identity:
            args.append('Name=key-name,Values={}'.format(self.identity))

        output = subprocess.check_output(args)
        parsed_out = json.loads(output)

        # flatten output
        instances = [instance
                     for reservation in parsed_out['Reservations']
                     for instance in reservation['Instances']]

        def sort_key_of_instance(instance):
            # type: (Dict[str, Any]) -> Tuple[time.struct_time, str]
            return (time.strptime(instance['LaunchTime'], '%Y-%m-%dT%H:%M:%S.%fZ'), instance['InstanceId'])

        return sorted(instances, key=sort_key_of_instance)
