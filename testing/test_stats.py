import pytest
import os
import subprocess
import glob
from conftest import *
import common_libs.utilities as ut
import mysql.connector


@ithemal
class TestStats:

    def test_getbenchmarks(self):

        script = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/stats/getbenchmarks.py'
        database = '--database=testIthemal'
        config = '--config=test_data/db_config.cfg'
        arch = '--arch=1'

        args = ['python',script, database, config, arch]
        proc = subprocess.Popen(args,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout, stderr = proc.communicate()


        success = False
        for line in stdout.split('\n'):
            if line == 'Total 44 2934 0':
                success = True

        assert success

