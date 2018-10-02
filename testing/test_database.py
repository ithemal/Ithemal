import pytest
import os
import subprocess
import glob
from conftest import *
import common_libs.utilities as ut

@dynamorio
class TestDatabase:


    def test_connectivity():
        
        
