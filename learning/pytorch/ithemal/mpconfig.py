import os
from typing import Any, Optional, Tuple

class MPConfig(object):
    THREADS_KEY = "OMP_NUM_THREADS"
    AFFINITY_KEY = "KMP_AFFINITY"

    # PyTorch starts 2 of its own threads for each trainer, so we actually want to start 2 fewer threads
    PYTORCH_THREAD_OFFSET = 2


    def __init__(self, threads):
        # type: (int) -> None
        assert 2 <= threads

        self.threads = threads
        self.saved_env = None # type: Optional[Tuple[Optional[str], Optional[str]]]

    def __enter__(self):
        # type: () -> None
        threads = os.environ.get(MPConfig.THREADS_KEY)
        affinity = os.environ.get(MPConfig.AFFINITY_KEY)

        self.saved_env = (threads, affinity)

    def set_env(self, trainer_id):
        # type: (int) -> None

        # set the OMP config, to get threads on sequential CPUs, ideally on the same socket
        os.environ[MPConfig.THREADS_KEY] = str(self.threads - MPConfig.PYTORCH_THREAD_OFFSET)
        os.environ[MPConfig.AFFINITY_KEY] = ','.join(map(str, [
            'verbose',
            'granularity=fine',
            'compact',
            '1',
            trainer_id * self.threads
        ]))

    def __exit__(self,exc_type, exc_value, traceback):
        # type: (Any, Any, Any) -> None
        assert self.saved_env

        (threads, affinity) = self.saved_env

        if threads is not None:
            os.environ[MPConfig.THREADS_KEY] = threads

        if affinity is not None:
            os.environ[MPConfig.AFFINITY_KEY] = affinity

        self.saved_env = None
