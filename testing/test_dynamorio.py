import pytest
import os
import subprocess

dynamorio = pytest.mark.skipif('DYNAMORIO_HOME' not in os.environ.keys(),
                                reason="DYNAMORIO_HOME not set")
@dynamorio
class TestDynamoRIO:

    def test_dynamorio_installation(self):
        
        drexec = os.environ['DYNAMORIO_HOME'] + '/bin64/drrun'
        proc = subprocess.Popen([drexec,'--','ls'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = proc.communicate()
        drout = stdout.split('\n')
        proc = subprocess.Popen(['ls'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = proc.communicate()
        normalout = stdout.split('\n')
            
        assert drout == normalout

    
