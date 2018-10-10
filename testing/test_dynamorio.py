import pytest
import os
import subprocess
import glob
from conftest import *


@dynamorio
class TestDynamoRIO:

    drexec = os.environ['DYNAMORIO_HOME'] + '/bin64/drrun'

    def test_dynamorio_installation(self):

        proc = subprocess.Popen([TestDynamoRIO.drexec,'--','ls'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        drout, _ = proc.communicate()
        proc = subprocess.Popen(['ls'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        normalout, _ = proc.communicate()

        assert drout == normalout

    @ithemal
    def test_drclient_static(self):

        static_client = os.environ['ITHEMAL_HOME'] + '/data_collection/build/bin/libstatic.so'

        files = glob.glob('/tmp/static_*')
        cmd = ['rm','-f'] + files
        proc = subprocess.Popen(cmd)
        proc.communicate()
        proc = subprocess.Popen([TestDynamoRIO.drexec,'-c',static_client,'3','7','1','gcc','none','/tmp','--','ls'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        drout, _ = proc.communicate()
        proc = subprocess.Popen(['ls'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        normalout, _ = proc.communicate()

        assert drout == normalout
        assert len(glob.glob('/tmp/static_*')) == 1

    @ithemal
    def test_created_sql_file(self):

        files = glob.glob('/tmp/static_*')

        assert len(files) == 1

        #check one line of the SQL file to see if it makes sense
        with open(files[0],'r') as f:

            line = f.readline()
            assert "INSERT INTO config (compiler, flags, arch)" in line
            line = f.readline()
            assert "INSERT INTO code" in line
            line = f.readline()
            assert "UPDATE code SET" in line



