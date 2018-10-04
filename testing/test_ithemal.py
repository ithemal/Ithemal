import pytest
import os
import subprocess
import glob
from conftest import *
import common_libs.utilities as ut
import mysql.connector
import urllib

script = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/ithemal/run_ithemal.py'
database = '--database=costmodel'
config = '--config=test_data/db_config.cfg'
arch = '--arch=2'
        

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

    # def test_training(self):

    #     savedata = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/inputs/data/timing_skylake.data'
    #     args = ['python',script, arch, '--mode=train','--savedatafile=' + savedata]
    #     proc = subprocess.Popen(args,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    #     stdout, stderr = proc.communicate()

    #     print stdout
    #     success = False
    #     for line in stdout.split('\n'):
    #         if line == 'Total 44 2934 0':
    #             success = True
        
    #     assert success


