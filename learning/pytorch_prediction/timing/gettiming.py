
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import common.utilities as ut
from tqdm import tqdm
import subprocess
import os
import re
import time
import argparse


def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = min(seconds / 1000.0, .25)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            proc.kill()
            return None
        time.sleep(interval)

def remove_unrecog_words(line):

    words = ['ptr', '<rel>']

    for word in words:
        line = line.replace(word,'')
    return line

def add_memory_prefix(line):
    mem = re.search('.*\[(.*)\].*', line)
    if mem != None and 'rsp' not in line:
        index = mem.span(1)[0]
        line = line[:index] + 'UserData + ' + line[index:]
    return line


def update_time_value(cnx,code_id, time):

    sql = 'UPDATE code SET time=' + str(time) + ' WHERE code_id=' + str(code_id)
    ut.execute_query(cnx, sql, False)
    cnx.commit()
    

class PMCValue:
    
    def __init__(self, value):
        self.value = value
        self.count = 1

class PMC:

    def __init__(self, name):
        self.name = name
        self.values = []
        self.mode = None
        self.percentage = 10

    def add_value(self, nvalue):

        added = False
        for val in self.values:
            if val.value == 0:
                val.value = 1e-3
            if (abs(val.value - nvalue) * 100.0 / val.value)  < self.percentage:
                val.value = (val.value * val.count + nvalue) / (val.count + 1)
                val.count += 1
                added = True
                break

        if not added:
            val = PMCValue(nvalue)
            self.values.append(val)

    def set_mode(self):
        
        max_count = 0

        for val in self.values:
            if val.count > max_count:
                self.mode = val.value
                max_count = val.count

class PMCCounters:

    def __init__(self,line):
        names = line.split()
        #print names
        self.counters = list()
        for name in names:
            self.counters.append(PMC(name))
    
    def add_to_counters(self, line):
        values = line.split()
        #print values

        for i, value in enumerate(values):
            self.counters[i].add_value(int(value))

    def set_modes(self):
        
        for counter in self.counters:
            counter.set_mode()

    def get_value(self, name):
        
        for counter in self.counters:
            if name == counter.name:
                return counter.mode
        return None

def check_error(line):
    
    errors = ['error','fault']
    
    for error in errors:
        if error in line:
            return True
    return False

if __name__ == '__main__':

    cnx = ut.create_connection('static')
    
#command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean',action='store',default=False,type=bool)
    args = parser.parse_args(sys.argv[1:])


    sql = 'SELECT code_text, code_id from code'
    rows = ut.execute_query(cnx, sql, True)
    print len(rows)

    os.chdir('/data/scratch/charithm/projects/cmodel/agner/testp/PMCTest')
    
    lines = []
    start_line = -1
    with open('PMCTestB64.nasm','r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            rep = re.search('.*%REP.*', line)
            if rep != None:
                start_line = i
                break
    
    print start_line

    total = 0
    errors = 0
    except_errors = 0
    success = 0
    not_finished = 0


    #clean all previous timing values
    if args.clean:
        print 'cleaning data....'
        for row in tqdm(rows):
            sql = 'UPDATE code SET time=NULL WHERE code_id=' + str(row[1])
            ut.execute_query(cnx, sql, False)
    
        cnx.commit()

    for row in rows:
    
        if row[0] == None:
            continue

        splitted = row[0].split('\n')
        write_lines = [line for line in lines]
        
        written = 0
        final_bb = []
        for i, line in enumerate(splitted):
            if line != '':
                line = remove_unrecog_words(line + '\n')
                line = add_memory_prefix(line)
                final_bb.append(line)
                write_lines.insert(start_line + 1 + i, line)
                written += 1

        #for line in final_bb:
        #    print line[:-1]


        #written = 1
        if written > 0:
            total += 1
            with open('out.nasm','w+') as f:
                f.writelines(write_lines)
            proc = subprocess.Popen('./a64-out.sh', stdout=subprocess.PIPE, stderr=subprocess.PIPE)            
            result = wait_timeout(proc, 10)

            
            if result != None:
                
                try:
                    error_lines = False
                    for line in iter(proc.stderr.readline, ''):
                        print line
                        if check_error(line):
                            error_lines = True
                            break
 
                    if error_lines == False:
                        startHeading = False
                        startTimes = False
                        counters = None
                        for i, line in enumerate(iter(proc.stdout.readline, '')):
                            #print line
                            if 'Clock' in line and startTimes == False and startHeading == False: #still didn't start collecting the actual timing data
                                startHeading = True
                            if startHeading == True:
                                #print 'headings ' + line
                                counters = PMCCounters(line)
                                startTimes = True
                                startHeading = False
                            elif startTimes == True:
                                #print 'values ' + line
                                counters.add_to_counters(line)
                        if counters != None:
                            counters.set_modes()
                            mode = counters.get_value('Core_cyc')
                            print mode
                            if mode != None:                    
                                update_time_value(cnx, row[1], mode)
                                success += 1
                    else:
                        for line in final_bb:
                            print line[:-1]
                        errors += 1
                except:
                    print 'exception occurred'
                    except_errors += 1

            else:
                print 'error not completed'
                not_finished += 1
        
        print total, success, errors, not_finished, except_errors

    cnx.close()