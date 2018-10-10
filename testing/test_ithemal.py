import pytest
import os
import subprocess
import glob
from conftest import *
import common_libs.utilities as ut
import mysql.connector
import urllib
import time

database = '--database=test_costmodel'
config = '--config=test_data/db_config.cfg'
arch = '--arch=2'

home = os.environ['ITHEMAL_HOME']
script = home + '/learning/pytorch/ithemal/run_ithemal.py'
savedata = home + '/learning/pytorch/inputs/data/time_skylake_test.data'
embedfile = home + '/learning/pytorch/inputs/embeddings/code_delim.emb'
savemodel = home + '/learning/pytorch/inputs/models/test_skylake.mdl'


def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = 30

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            proc.kill()
            return None
        time.sleep(interval)


@ithemal
class TestIthemal:

    def test_create_ithemal_database(self):

        urllib.urlretrieve ("http://web.mit.edu/charithm/www/test_costmodel.sql", "test_data/test_costmodel.sql")
        assert os.path.exists('test_data/test_costmodel.sql')

        default_file = 'test_data/db_config.cfg'
        cnx = ut.create_connection_from_config(default_file)
        assert cnx

        ut.execute_query(cnx,'drop database if exists test_costmodel',False)
        cnx_none = ut.create_connection_from_config(default_file,'test_costmodel')
        assert cnx_none == None

        ut.execute_query(cnx,'create database if not exists test_costmodel',False)
        cnx.close()

        cnx = ut.create_connection_from_config(default_file,'test_costmodel')
        assert cnx

        sql = open('test_data/test_costmodel.sql').read()

        for line in sql.split(';'):
            print line
            ut.execute_query(cnx,line,False,True)
        cnx.commit()

        rows = ut.execute_query(cnx,'select count(*) from code',True)
        assert rows[0][0] == 100000

    def test_savedata(self):


        args = ['python',script, config, database, arch, '--mode=save','--savedatafile=' + savedata]
        proc = subprocess.Popen(args,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout, stderr = proc.communicate()

        print stdout
        success = False
        for line in stdout.split('\n'):
            if line == 'timing values registered for 78179 items':
                success = True

        assert success

    @pytest.mark.skip
    def test_training(self):

        args = ['python',script, '--mode=train','--savedatafile=' + savedata, '--savefile=' + savemodel, '--embedfile=' + embedfile, '--embmode=none']

        proc = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        wait_timeout(proc,300)

        output = []
        for line in proc.stdout:
            output.append(line.decode())

        print("".join(output))

        assert False


