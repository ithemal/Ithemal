#!/usr/bin/env python

import argparse
import atexit
import aws_utils.queue_process
import curses
import json
import os
import start_instance
import subprocess
import sys
try:
    import urlparse
except ImportError:
    import urrlib.parse as urlparse
from typing import Optional
import tempfile
from typing import Any, Dict, List

# Ithemal runs on Python 2 mostly
try:
    input = raw_input
except NameError:
    pass

_DIRNAME = os.path.dirname(os.path.abspath(__file__))

def queue_url_of_name(queue_name):
    # type: (str) -> Optional[str]
    proc = subprocess.Popen(
        ['aws', 'sqs', 'get-queue-url', '--queue-name', queue_name + '.fifo'],
        stdout=subprocess.PIPE,
        stderr=open('/dev/null', 'w'),
    )
    if proc.wait() or not proc.stdout:
        return None

    output = json.load(proc.stdout)
    return output['QueueUrl']

def create_queue(identity, queue, instance_type, instance_count, ignore_exists, no_spot):
    # type: (str, str, str, int, bool, bool) -> None

    queue_exists = queue_url_of_name(queue)
    if queue_exists and not ignore_exists:
        print('Queue {} already exists!'.format(queue))
        return

    queue_url = json.loads(subprocess.check_output([
        'aws', 'sqs', 'create-queue',
        '--queue-name', queue + '.fifo',
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
                identity, '-f', '--no-connect',
                '-t', instance_type,
                '--name', '{} Queue Processor'.format(queue),
                '--queue', queue,
            ] + ([] if no_spot else ['--spot-preempt']),
            stdout=stdout,
            stderr=stderr,
        ))

    try:
        for proc in procs:
            proc.wait()
    except (KeyboardInterrupt, SystemExit):
        for proc in procs:
            proc.terminate()
        if not queue_exists:
            kill_queue(queue)

def send_messages(queue, com):
    # type: (str, str) -> None

    url = queue_url_of_name(queue)
    if not url:
        print('Queue {} doesn\'t exist!'.format(queue))
        return

    if com:
        aws_utils.queue_process.send_message(url, ' '.join(com))
    else:
        try:
            while True:
                if sys.stdin.isatty():
                    com = input('com> ')
                else:
                    com = input()
                aws_utils.queue_process.send_message(url, com)
        except (EOFError, KeyboardInterrupt):
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

def running_of_queue(identity, queue):
    # type: (str, str) -> None

    def has_queue_tag(instance):
        # type: (Dict[str, Any]) -> bool

        if 'Tags' not in instance:
            return False

        for tag in instance['Tags']:
            if tag['Key'] == 'QueueName' and tag['Value'] == queue:
                return True
        return False

    instances_json = json.loads(subprocess.check_output(['aws', 'ec2', 'describe-instances', '--filters', 'Name=instance-state-name,Values=pending,running']))
    instances = [i for res in instances_json['Reservations'] for i in res['Instances'] if has_queue_tag(i)]

    for instance in instances:
        out = subprocess.check_output([
            os.path.join(_DIRNAME, 'connect_instance.py'), identity, instance['InstanceId'],
            '--com', os.path.join('${ITHEMAL_HOME}', 'aws', 'aws_utils', 'get_running_queue_command.sh')
        ], stderr=open('/dev/null', 'w')).strip()
        if out:
            print('{} || {}'.format(instance['InstanceId'], out))


def preview_queue(queue):
    # type: (str) -> None

    url = queue_url_of_name(queue)
    if not url:
        print('Queue {} doesn\'t exist!'.format(queue))
        return

    output = subprocess.check_output([
        'aws', 'sqs', 'receive-message',
        '--queue-url', url,
        '--visibility-timeout', '0',
        '--max-number-of-messages', '10',
    ])

    if not output:
        print('No messages in queue!')
        return

    messages = json.loads(output)['Messages']

    for message in messages:
        print('> {}'.format(message['Body']))

def manage_queue(queue):
    # type: (str) -> None

    url_ = queue_url_of_name(queue)
    if not url_:
        print('Queue {} doesn\'t exist!'.format(queue))
        return
    else:
        url = url_

    messages = [] # type: List[Dict[str, Any]]

    def reset_messages():
        # type: () -> None
        if not messages:
            return

        with tempfile.NamedTemporaryFile(suffix='.json', bufsize=0) as f:
            json.dump([{
                'Id': message['MessageId'],
                'ReceiptHandle': message['ReceiptHandle'],
                'VisibilityTimeout': 0,
            } for message in messages], f)

            output = subprocess.check_output([
                'aws', 'sqs', 'change-message-visibility-batch',
                '--queue-url', url,
                '--entries', 'file://{}'.format(f.name),
            ])

    def get_messages():
        # type: () -> List[Dict[str, Any]]
        reset_messages()

        output = subprocess.check_output([
            'aws', 'sqs', 'receive-message',
            '--queue-url', url,
            '--visibility-timeout', '30',
            '--max-number-of-messages', '10',
        ])

        if not output:
            return []
        else:
            return json.loads(output)['Messages']

    messages = get_messages()

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(1)

    def cleanup():
        # type: () -> None
        curses.nocbreak()
        stdscr.keypad(0);
        curses.echo()
        curses.endwin()
        reset_messages()

    atexit.register(cleanup)

    selected_idx = 0

    while True:
        if not messages:
            return

        stdscr.erase()
        stdscr.addstr(0, 0, 'Queue Management mode. Arrow keys to move, x to delete an item, q to exit', curses.A_STANDOUT)
        stdscr.addstr(selected_idx + 2, 0, '>')

        (maxy, maxx) = stdscr.getmaxyx()
        for i, message in enumerate(messages):
            body = message['Body'][:maxx-5] + (message['Body'][maxx-5:] and '...')
            stdscr.addstr(i + 2, 2, body)
        stdscr.refresh()

        c = stdscr.getch()
        if c == ord('x'):
            message = messages[selected_idx]

            stdscr.erase()
            stdscr.addstr(0, 0, 'Really delete the following item? (y/n)')
            stdscr.addstr(1, 0, message['Body'])
            stdscr.refresh()

            while True:
                c = stdscr.getch()
                if c == ord('y'):
                    subprocess.check_call(['aws', 'sqs', 'delete-message', '--queue-url', url, '--receipt-handle', message['ReceiptHandle']])
                    selected_idx = 0
                    break
                elif c == ord('n'):
                    break

            messages = get_messages()
        elif c == ord('q'):
            break
        elif c == curses.KEY_DOWN:
            selected_idx = min(selected_idx + 1, len(messages) - 1)
        elif c == curses.KEY_UP:
            selected_idx = max(selected_idx - 1, 0)

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
            name = full_name[:-len(suffix)]
        else:
            name = full_name

        return '{} ({})'.format(name, url)

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
    create_parser.add_argument('--ignore-exists', help='Fork instances regardless of if the queue exists or not', action='store_true', default=False)
    create_parser.add_argument('--no-spot', help='Start on-demand instead of spot instances', action='store_true', default=False)

    send_parser = subparsers.add_parser('send', help='Send messages to AWS queues')
    add_queue_arg(send_parser)
    send_parser.add_argument('com', nargs='*', help='Command to send (if empty, read lines from stdin)')

    kill_parser = subparsers.add_parser('kill', help='Kill AWS queue')
    add_queue_arg(kill_parser)

    preview_parser = subparsers.add_parser('preview', help='Preview an AWS queue')
    add_queue_arg(preview_parser)

    manage_parser = subparsers.add_parser('manage', help='Manage an AWS queue')
    add_queue_arg(manage_parser)

    running_parser = subparsers.add_parser('running', help='Get commands currently running on an AWS queue')
    running_parser.add_argument('identity', help='Identity to use to connect')
    add_queue_arg(running_parser)

    args = parser.parse_args()

    if args.subparser == 'list':
        list_queues()
        return

    if args.subparser == 'create':
        create_queue(args.identity, args.queue_name, args.type, args.count, args.ignore_exists, args.no_spot)
    elif args.subparser == 'send':
        send_messages(args.queue_name, args.com)
    elif args.subparser == 'kill':
        kill_queue(args.queue_name)
    elif args.subparser == 'preview':
        preview_queue(args.queue_name)
    elif args.subparser == 'manage':
        manage_queue(args.queue_name)
    elif args.subparser == 'running':
        running_of_queue(args.identity, args.queue_name)
    else:
        raise ValueError('Unrecognized subparser {}'.format(args.subparser))

if __name__ == '__main__':
    main()
