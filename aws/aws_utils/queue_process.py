#!/usr/bin/env python

import argparse
import json
import os
import requests
import spot_checker
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from typing import NoReturn, Optional

def send_message(queue_url, com):
    # type: (str, str) -> None
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
    # type: (str) -> None
    subprocess.call(['aws', 'ec2', 'terminate-instances', '--instance-ids', instance_id])
    sys.exit(1)

curr_com = None # type: Optional[str]

def watch_for_instance_death(queue_url, instance_id):
    # type: (str, str) -> None
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

        death_message = ':skull_and_crossbones: Spot instance {} dying :skull_and_crossbones:'.format(instance_id)

        if curr_com:
            send_message(queue_url, curr_com)
            death_message += '\nRe-queueing {}'.format(curr_com)

        subprocess.call([os.path.join(os.environ['ITHEMAL_HOME'], 'aws', 'ping_slack.py'), death_message])
        return

def process_queue(instance_id, queue_url, kill_on_fail):
    # type: (str, str, bool) -> None
    global curr_com

    t = threading.Thread(target=watch_for_instance_death, args=(queue_url, instance_id))
    t.daemon = True
    t.start()

    log_file = open('/tmp/queue_log', 'a+', 1)

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
        com = message['Body']

        mk_tmp_file = lambda suffix: open(os.path.join('/tmp', '{}.{}'.format(
            message['MessageId'],
            suffix
        )), 'w')

        with mk_tmp_file('stdout') as stdout, mk_tmp_file('stderr') as stderr:
            log_file.write(com + '\n')
            log_file.write('--> stdout: {}, stderr: {}\n\n'.format(stdout.name, stderr.name))
            proc = subprocess.Popen(
                message['Body'],
                shell=True,
                cwd=os.environ['ITHEMAL_HOME'],
                stdout=stdout,
                stderr=stderr
            )
            proc.wait()

            if not proc.returncode:
                # if process executed successfully, delete stderr file
                os.unlink(stderr.name)

        curr_com = None

        if proc.returncode:
            error_msg = 'Command `{}` failed with exit code {} on instance {}'.format(message['Body'],  proc.returncode, instance_id)
            subprocess.call([os.path.join(os.environ['ITHEMAL_HOME'], 'aws', 'ping_slack.py'), error_msg])

def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Indefinitely pull messages from a given AWS queue')
    parser.add_argument('queue_url', help='The AWS SQS queue URL to pull from')
    parser.add_argument('--kill', help='Kill the instance if a queue pull fails', action='store_true', default=False)
    args = parser.parse_args()

    instance_id = requests.get('http://169.254.169.254/latest/meta-data/instance-id').text

    try:
        process_queue(instance_id, args.queue_url, args.kill)
    except:
        error_msg = 'Error on instance {}:\n```{}```'.format(instance_id, traceback.format_exc())
        subprocess.call([os.path.join(os.environ['ITHEMAL_HOME'], 'aws', 'ping_slack.py'), error_msg])

if __name__ == '__main__':
    main()
