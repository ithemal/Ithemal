import mysql.connector
import struct
import sys
from mysql.connector import errorcode
import re
import os

#strategy 1
#load basic blocks from the database
#define operand kinds - arithmetic, movs etc. and create basic blocks with mulitple of these instructions (independent)

#strategy 2
#create basic blocks from scratch
#problem : we may not perform the correct tokenization

#time the basic blocks and insert them into a separate database
#run it through Ithemal to see if it learns


