#!/usr/bin/python

import argparse
import json
import subprocess
import urllib2
import os

WEBHOOK_URL = 'https://hooks.slack.com/services/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'
SLACK_USERNAME = 'AWS'
SLACK_CHANNEL = 'aws-notifications'
SLACK_ICON = 'https://raw.githubusercontent.com/quintessence/slack-icons/e9e141f0a119759ca4d59e0b788fc9375c9b2678/images/amazon-web-services-slack-icon.png'

USER_MAP = {
    'renda': 'UCJ98TMB8',
    'charithm': 'UB59J5BHR',
    'mcarbin': 'U7QK3FX88',
}

def get_starting_user():
    proc = subprocess.Popen(
        ['/usr/bin/curl', '--silent', '--connect-timeout', '1', 'http://169.254.169.254/latest/meta-data/public-keys/0/openssh-key'],
        stdout=subprocess.PIPE,
        stderr=open(os.devnull, 'w'),
    )
    proc.wait()
    if proc.returncode:
        return None

    stdout, _ = proc.communicate()
    stdout = stdout.strip()
    return stdout.split()[2]

def send_message(message):
    payload = {
        'text': message,
        'username': SLACK_USERNAME,
        'icon_url': SLACK_ICON,
        'channel': SLACK_CHANNEL,
    }

    request = urllib2.Request(WEBHOOK_URL, json.dumps(payload))
    urllib2.urlopen(request)

def main():
    parser = argparse.ArgumentParser('Ping a user in the aws-notifications channel on Slack')
    parser.add_argument('--user', default=None, help='User to ping (default: user that started instance on AWS, None elsewhere)')
    parser.add_argument('message', help='Message to send')

    args, unknown_args = parser.parse_known_args()

    message = args.message
    for unknown_arg in unknown_args:
        message += ' ' + unknown_arg

    if args.user:
        user = args.user
    else:
        user = get_starting_user()

    if user:
        message = '<@{}>: {}'.format(
            USER_MAP[user],
            message,
        )

    send_message(message)

if __name__ == '__main__':
    main()
