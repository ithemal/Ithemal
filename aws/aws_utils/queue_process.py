#!/usr/bin/env python

import argparse
import json
import os
import requests
import subprocess
import sys

def kill_instance(instance_id):
    subprocess.call(['aws', 'ec2', 'terminate-instances', '--instance-ids', instance_id])
    sys.exit(1)

def process_queue(queue_url):
    instance_id = requests.get('http://169.254.169.254/latest/meta-data/instance-id').text

    while True:
        try:
            output = subprocess.check_output(['aws', 'sqs', 'receive-message', '--queue-url', queue_url, '--wait-time-seconds', '20'])
        except subprocess.CalledProcessError:
            kill_instance(instance_id)

        if not output:
            continue

        messages = json.loads(output)
        message = messages['Messages'][0]
        subprocess.check_call(['aws', 'sqs', 'delete-message', '--queue-url', queue_url, '--receipt-handle', message['ReceiptHandle']])

        code = subprocess.call(message['Body'], shell=True, cwd=os.environ['ITHEMAL_HOME'])
        if code:
            subprocess.call([os.path.join(os.environ['ITHEMAL_HOME'], 'aws', 'ping_slack.py'),
                             'Command {} failed with exit code {} on instance {}'.format(code, instance_id)])

def main():
    parser = argparse.ArgumentParser('Indefinitely pull messages from a given AWS queue')
    parser.add_argument('queue_url', help='The AWS SQS queue URL to pull from')
    args = parser.parse_args()

    process_queue(args.queue_url)

if __name__ == '__main__':
    main()
