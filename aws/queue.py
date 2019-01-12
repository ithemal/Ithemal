#!/usr/bin/env python

import argparse
from aws_utils.queue_process import send_message
import json
import os
import start_instance
import subprocess
import sys
import urlparse
from typing import Optional

# Ithemal runs on Python 2 mostly
try:
    input = raw_input
except NameError:
    pass

_DIRNAME = os.path.dirname(os.path.abspath(__file__))

def queue_url_of_name(queue_name):
    # type: (str) -> Optional[str]
    proc = subprocess.Popen(
        ['aws', 'sqs', 'get-queue-url', '--queue-name', queue_name],
        stdout=subprocess.PIPE,
        stderr=open('/dev/null', 'w'),
    )
    if proc.wait() or not proc.stdout:
        return None

    output = json.load(proc.stdout)
    return output['QueueUrl']

def create_queue(identity, queue, instance_type, instance_count):
    # type: (str, str, str, int) -> None

    if queue_url_of_name(queue):
        print('Queue {} already exists!'.format(queue))
        return

    queue_url = json.loads(subprocess.check_output([
        'aws', 'sqs', 'create-queue',
        '--queue-name', queue,
        '--attributes', json.dumps({'FifoQueue': 'true'}),
    ]))['QueueUrl']

    procs = []
    for idx in range(instance_count):
        # keep outputs from last process to get an idea of spot instance stability
        if idx == instance_count - 1:
            stdout = None
            stderr = None
        else:
            stdout = open('/dev/null', 'w')
            stderr = open('/dev/null', 'w')

        procs.append(subprocess.Popen(
            [
                os.path.join(_DIRNAME, 'start_instance.py'),
                identity, '-f', '--spot-preempt', '--no-connect',
                '-t', instance_type,
                '--name', '{} Queue Processor'.format(queue),
                '--queue', queue_url,
            ],
            stdout=stdout,
            stderr=stderr,
        ))

    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        kill_queue(queue)

def send_messages(queue, com):
    # type: (str, str) -> None

    url = queue_url_of_name(queue)
    if not url:
        print('Queue {} doesn\'t exist!'.format(queue))
        return

    if com:
        send_message(url, ' '.join(com))
    else:
        try:
            while True:
                if sys.stdin.isatty():
                    com = input('com> ')
                else:
                    com = input()
                send_message(url, com)
        except EOFError, KeyboardInterrupt:
            pass

def kill_queue(queue):
    # type: (str) -> None

    url = queue_url_of_name(queue)
    if not url:
        print('Queue {} doesn\'t exist!'.format(queue))
        return

    subprocess.check_call([
        'aws', 'sqs', 'delete-queue',
        '--queue-url', url,
    ])

def preview_queue(queue):
    # type: (str) -> None

    url = queue_url_of_name(queue)
    if not url:
        print('Queue {} doesn\'t exist!'.format(queue))
        return

    output = subprocess.check_output([
        'aws', 'sqs', 'receive-message',
        '--queue-url', url,
        '--max-number-of-messages', '10',
        '--visibility-timeout', '0',
    ])

    if not output:
        print('No messages in queue!')
        return

    messages = json.loads(output)['Messages']

    for message in messages:
        print('> {}'.format(message['Body']))

def list_queues():
    # type: () -> None
    output = subprocess.check_output(['aws', 'sqs', 'list-queues'])
    if not output:
        print('No running queues!')
        return

    queues = json.loads(output)['QueueUrls']

    def parse_url(url):
        # type: (str) -> str
        full_name = urlparse.urlparse(url).path.split('/')[-1]
        suffix = '.fifo'
        if full_name.endswith(suffix):
            return full_name[:-len(suffix)]
        else:
            return full_name

    print('\n'.join(map(parse_url, queues)))

def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description='Manage AWS SQS queues and their associated workers')

    def add_queue_arg(sp):
        # type: (argparse.ArgumentParser) -> None
        sp.add_argument('queue_name', help='Queue to manage')

    subparsers = parser.add_subparsers(dest='subparser')

    list_parser = subparsers.add_parser('list', help='List AWS queues')

    create_parser = subparsers.add_parser('create', help='Create AWS queues')
    create_parser.add_argument('identity', help='Identity to use to connect')
    add_queue_arg(create_parser)
    create_parser.add_argument('-c', '--count', help='Number of queue processors to create', default=4, type=int)
    create_parser.add_argument('-t', '--type', help='Instance type to start (default: t2.large)', default='t2.large')

    send_parser = subparsers.add_parser('send', help='Send messages to AWS queues')
    add_queue_arg(send_parser)
    send_parser.add_argument('com', nargs='*', help='Command to send (if empty, read lines from stdin)')

    kill_parser = subparsers.add_parser('kill', help='Kill AWS queue')
    add_queue_arg(kill_parser)

    preview_parser = subparsers.add_parser('preview', help='Preview an AWS queue')
    add_queue_arg(preview_parser)

    args = parser.parse_args()

    if args.subparser == 'list':
        list_queues()
        return

    queue_name = args.queue_name
    if not queue_name.endswith('.fifo'):
        queue_name += '.fifo'

    if args.subparser == 'create':
        create_queue(args.identity, queue_name, args.type, args.count)
    elif args.subparser == 'send':
        send_messages(queue_name, args.com)
    elif args.subparser == 'kill':
        kill_queue(queue_name)
    elif args.subparser == 'preview':
        preview_queue(queue_name)
    else:
        raise ValueError('Unrecognized subparser {}'.format(args.subparser))

if __name__ == '__main__':
    main()
