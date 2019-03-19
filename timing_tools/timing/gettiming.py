from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import common_libs.utilities as ut
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

def fix_reg_names(line):
    # nasm recognizes, for instance, r14d rather than r14l
    regs = [('r%dl'%x, 'r%dd'%x) for x in range(8, 16)]
    for old, new in regs:
        line = line.replace(old, new)
    return line

def remove_unrecog_words(line):

    words = ['ptr', '<rel>']

    for word in words:
        line = line.replace(word,'')
    return line

def add_memory_prefix(line):
    mem = re.search('.*\[(.*)\].*', line)
    if (mem != None and
        re.match('.*(rsp|rbp|esp|ebp)', mem.group(1)) is None and
        not line.strip().startswith('lea')):
        index = mem.span(1)[0]
        line = line[:index] + 'UserData + ' + line[index:]
    return line


def insert_time_value(cnx,code_id, time, arch, ttable):

    sql = 'INSERT INTO ' + ttable + ' (code_id, arch, kind, time) VALUES(' + str(code_id) + ',' + str(arch) + ',\'actual\',' + str(time) + ')'
    ut.execute_query(cnx, sql, False)
    cnx.commit()

def insert_col_values(cnx, cols, values, code_id, arch, ttable):

    for i in range(len(values[0])):
        
        colstr = ''
        valuestr = ''

        for j, col in enumerate(cols): 
            if j != len(cols) - 1:
                colstr += col + ', '
                valuestr += str(values[j][i]) + ', '
            else:
                colstr += col
                valuestr += str(values[j][i])
                

        sql = 'INSERT INTO ' + ttable + ' (code_id, arch, kind,' + colstr + ')  VALUES(' + str(code_id) + ',' + str(arch) + ',\'actual\',' + valuestr + ')'
        print sql
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

        self.mod_values = []
        self.mode = None
        self.percentage = 5

    def add_value(self, nvalue):

        self.values.append(nvalue)

        added = False
        for val in self.mod_values:
            if val.value == 0:
                val.value = 1e-3
            if (abs(val.value - nvalue) * 100.0 / val.value)  < self.percentage:
                val.value = (val.value * val.count + nvalue) / (val.count + 1)
                val.count += 1
                added = True
                break

        if not added:
            val = PMCValue(nvalue)
            self.mod_values.append(val)
        
    def set_mode(self):

        max_count = 0

        for val in self.mod_values:
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

        if len(values) != len(self.counters):
            return

        for i, value in enumerate(values):
            self.counters[i].add_value(int(value))

    def set_modes(self):

        for counter in self.counters:
            counter.set_mode()

    def get_value(self, name):

        for counter in self.counters:
            if name == counter.name:
                return counter.values
        return None

    def get_mode(self, name):

        for counter in self.counters:
            if name == counter.name:
                return counter.mode
        return None

def check_error(line):

    errors = ['error','fault']
    warnings = ['warning']

    for error in errors:
        for warning in warnings:
            if error in line and not warning in line:
                return True
    return False

if __name__ == '__main__':


    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',action='store',type=int,required=True)

    parser.add_argument('--database',action='store',type=str,required=True)
    parser.add_argument('--user',action='store', type=str, required=True)
    parser.add_argument('--password',action='store', type=str, required=True)
    parser.add_argument('--port',action='store', type=int, required=True)
    parser.add_argument('--ctable',action='store',type=str, required=True)
    parser.add_argument('--ttable',action='store',type=str, required=True)
    parser.add_argument('--limit',action='store',type=int, default=None)
    parser.add_argument('--tp',action='store',type=bool,default=False)

    args = parser.parse_args(sys.argv[1:])

    cnx = ut.create_connection(database=args.database, user=args.user, password=args.password, port=args.port)
    sql = 'SELECT code_intel, code_id from ' + args.ctable
    rows = ut.execute_query(cnx, sql, True)
    print len(rows)

    harness_dir = os.environ['ITHEMAL_HOME'] + '/timing_tools/harness'
    os.chdir(harness_dir)

    total = 0
    errors = 0
    except_errors = 0
    success = 0
    not_finished = 0


    total_time = 0.0
    total_bbs = 0

    # do a dry run to figure out measurement overhead
    with open('bb.nasm', 'w') as f:
      f.close()
    proc = subprocess.Popen('./a64-out.sh', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = wait_timeout(proc, 10)
    startHeading = False
    startTimes = False
    counters = None
    for i, line in enumerate(iter(proc.stdout.readline, '')):
        if 'Clock' in line and startTimes == False and startHeading == False: #still didn't start collecting the actual timing data
            startHeading = True
        if startHeading == True:
            counters = PMCCounters(line)
            startTimes = True
            startHeading = False
        elif startTimes == True:
            counters.add_to_counters(line)
    assert counters is not None
    counters.set_modes()
    overhead = counters.get_mode('Core_cyc')
    print 'OVERHEAD =', overhead

    for row in rows:

        if row[0] == None:
            continue

        splitted = row[0].split('\n')

        written = 0
        final_bb = []
        for i, line in enumerate(splitted):
            if line != '':
                line = remove_unrecog_words(line + '\n')
                line = fix_reg_names(line)
                final_bb.append(line)
                written += 1



        if written > 0:
            total += 1
            with open('bb.nasm','w+') as f:
                f.writelines(final_bb)
            proc = subprocess.Popen('./a64-out.sh', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            start_time = time.time()
            result = wait_timeout(proc, 10)
            end_time = time.time()

            if result != None:

                print final_bb

                try:
                    error_lines = False
                    for line in iter(proc.stderr.readline, ''):
                        if check_error(line):
                            print 'error ' + line
                            error_lines = True
                            break

                    if error_lines == False:
                        startHeading = False
                        startTimes = False
                        counters = None
                        for i, line in enumerate(iter(proc.stdout.readline, '')):
                            print line
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

                            names = ['Core_cyc', 'L1_read_misses', 'L1_write_misses', 'iCache_misses', 'Context_switches']
                            columns = ['time', 'l1drmisses', 'l1dwmisses', 'l1imisses', 'conswitch']

                            values = []
                            aval_cols = []

                            for i, name in enumerate(names):
                                vs = counters.get_value(name)
                                if vs != None:
                                    values.append(vs)
                                    aval_cols.append(columns[i])
                                    if name == 'Core_cyc':
                                        for j, v in enumerate(values[-1]):
                                            values[-1][j] -= overhead
                            print aval_cols, values

                            if not args.tp:
                                insert_col_values(cnx, aval_cols, values, row[1], args.arch, args.ttable)
                                    
                            total_time += end_time - start_time
                            total_bbs += 1
                            print float(total_bbs)/total_time
                            success += 1
                    else:
                        for line in final_bb:
                            print line[:-1]
                        errors += 1
                except Exception as e:
                    print e
                    print 'exception occurred'
                    except_errors += 1

            else:
                print 'error not completed'
                not_finished += 1

        if args.limit != None:
            if success == args.limit:
                break

        print total, success, errors, not_finished, except_errors


    print overhead
    cnx.close()
