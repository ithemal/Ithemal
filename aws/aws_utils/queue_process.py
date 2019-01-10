#!/usr/bin/env python

import argparse
import json
import os
import requests
import spot_checker
import subprocess
import sys
import threading
import time

def send_message(queue_url, com):
    ''' Utility to send a message directly to a URL without checking
    '''
    subprocess.check_call([
        'aws', 'sqs', 'send-message',
        '--queue-url', queue_url,
        '--message-body', com,
        '--message-group-id', 'none',
        '--message-deduplication-id', str(uuid.uuid4()),
    ])

def kill_instance(instance_id):
    subprocess.call(['aws', 'ec2', 'terminate-instances', '--instance-ids', instance_id])
    sys.exit(1)

curr_com = None

def watch_for_instance_death(queue_url, instance_id):
    global curr_com

    while True:
        death_time = spot_checker.get_termination_time()
        if not death_time:
            time.sleep(60)
            continue

        # sleep until 10s before instance termination
        sleep_dur = death_time - time.time() - 10
        if sleep_dur > 0:
            time.sleep(sleep_dur)

        if curr_com:
            send_message(queue_url, curr_com)

        subprocess.call([os.path.join(os.environ['ITHEMAL_HOME'], 'aws', 'ping_slack.py'),
                         ':skull_and_crossbones: Spot instance {} dying :skull_and_crossbones:'.format(instance_id)])

def process_queue(queue_url, kill_on_fail):
    global curr_com

    instance_id = requests.get('http://169.254.169.254/latest/meta-data/instance-id').text

    t = threading.Thread(target=watch_for_instance_death, args=(queue_url, instance_id))
    t.daemon = True
    t.start()

    while True:
        try:
            output = subprocess.check_output(['aws', 'sqs', 'receive-message', '--queue-url', queue_url, '--wait-time-seconds', '20'])
        except subprocess.CalledProcessError:
            if kill_on_fail:
                kill_instance(instance_id)
            else:
                return

        if not output:
            continue

        messages = json.loads(output)
        message = messages['Messages'][0]
        subprocess.check_call(['aws', 'sqs', 'delete-message', '--queue-url', queue_url, '--receipt-handle', message['ReceiptHandle']])

        curr_com = message['Body']
        code = subprocess.call(message['Body'], shell=True, cwd=os.environ['ITHEMAL_HOME'])
        curr_com = None

        if code:
            subprocess.call([os.path.join(os.environ['ITHEMAL_HOME'], 'aws', 'ping_slack.py'),
                             'Command {} failed with exit code {} on instance {}'.format(code, instance_id)])

def main():
    parser = argparse.ArgumentParser('Indefinitely pull messages from a given AWS queue')
    parser.add_argument('queue_url', help='The AWS SQS queue URL to pull from')
    parser.add_argument('--kill', help='Kill the instance if a queue pull fails', action='store_true', default=False)
    args = parser.parse_args()

    process_queue(args.queue_url, args.kill)

if __name__ == '__main__':
    main()
