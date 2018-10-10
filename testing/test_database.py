import pytest
import os
import subprocess
import glob
from conftest import *
import common_libs.utilities as ut
import mysql.connector


@ithemal
class TestDatabase:

    def test_connectivity(self,db_config):

        assert 'password' in db_config.keys()
        assert 'user' in db_config.keys()
        assert 'port' in db_config.keys()

        cnx = ut.create_connection(user=db_config['user'],password=db_config['password'],port=db_config['port'],database=None)
        assert cnx != None

    def test_connectivity_from_config(self):

        cnx = ut.create_connection_from_config('test_data/db_config.cfg')
        assert cnx != None

    def test_create_database(self,db_config):

        create_script = os.environ['ITHEMAL_HOME'] + '/data_export/scripts/create_and_populate_db.sh'
        schema = os.environ['ITHEMAL_HOME'] + '/data_export/schemas/mysql_schema.sql'

        proc = subprocess.call(['bash',create_script,'test_data/db_config.cfg','testIthemal',schema,'test_data'])
        #_ = proc.communicate()

        cnx = ut.create_connection(user=db_config['user'],password=db_config['password'],port=db_config['port'],database='testIthemal')
        assert cnx != None

        sql = 'select count(*) from code'

        rows = ut.execute_query(cnx, sql, True)

        assert len(rows) == 1
        assert len(rows[0]) == 1

        assert rows[0][0] == 3287

