import pytest
import os
import subprocess
import glob

dynamorio = pytest.mark.skipif('DYNAMORIO_HOME' not in os.environ.keys(),
                                reason="DYNAMORIO_HOME not set")

ithemal = pytest.mark.skipif('ITHEMAL_HOME' not in os.environ.keys(),
                                reason="ITHEMAL_HOME not set")
