import calendar
import requests
import time
from typing import Optional

def get_termination_time():
    # type: () -> Optional[float]
    resp = requests.get('http://169.254.169.254/latest/meta-data/spot/instance-action')
    if resp.status_code != 200:
        return None

    action = resp.json()

    if action['action'] == 'hibernate':
        return None

    return calendar.timegm(time.strptime(action['time'], '%Y-%m-%dT%H:%M:%SZ'))
