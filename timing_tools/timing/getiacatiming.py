from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import utilities as ut
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
    interval = min(seconds / 1000.0, .10)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            proc.kill()
            return None
        time.sleep(interval)


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
                return counter.mode
        return None


def insert_time_value(cnx,code_id, time, arch):

    sql = 'INSERT INTO times (code_id, arch, kind, time) VALUES(' + str(code_id) + ',' + str(arch) + ',\'iaca\',' + str(time) + ')'
    ut.execute_query(cnx, sql, False)
    cnx.commit()


def check_error(line):

    errors = ['error','fault','Error']

    for error in errors:
        if error in line:
            return True
    return False

if __name__ == '__main__':


    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',action='store',type=int,required=True)
    parser.add_argument('--cpu',action='store',type=str,required=True)

    parser.add_argument('--database',action='store',type=str,required=True)
    parser.add_argument('--user',action='store', type=str, required=True)
    parser.add_argument('--password',action='store', type=str, required=True)
    parser.add_argument('--port',action='store', type=int, required=True)

    parser.add_argument('--subd',action='store',type=str,default='')
    parser.add_argument('--tp',action='store',type=bool,default=False)
    parser.add_argument('--start',action='store',type=int)
    parser.add_argument('--end',action='store',type=int)

    args = parser.parse_args(sys.argv[1:])

    cnx = ut.create_connection(database=args.database, user=args.user, password=args.password, port=args.port)
    sql = 'SELECT code_att, code_id from code'
    rows = ut.execute_query(cnx, sql, True)
    print len(rows)

    iaca_home = os.environ['ITHEMAL_HOME'] + '/timing_tools/iaca/'
    os.chdir(iaca_home + args.subd)

    lines = []
    start_line = -1
    with open('test.s','r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            rep = re.search('.*\.rept.*', line)
            if rep != None:
                start_line = i
                break

    print start_line

    total = 0
    errors = 0
    except_errors = 0
    success = 0
    not_finished = 0

    total_time = 0


    start = int(len(rows) * 0.8)

    for row in tqdm(rows[start:]):

        if row[0] == None:
            continue

        if args.start and args.end:
            if row[1] < args.start or row[1] > args.end:
                continue

        splitted = row[0].split('\n')
        write_lines = [line for line in lines]

        written = 0
        final_bb = []
        for i, line in enumerate(splitted):
            if line != '':
                final_bb.append(line + '\n')
                write_lines.insert(start_line + 1 + i, line + '\n')
                written += 1

        #for line in final_bb:
        #    print line

        #written = 1
        if written > 0:
            total += 1
            with open('out.s','w+') as f:
                f.writelines(write_lines)
            proc = subprocess.Popen(['gcc','-c','-o','test.o','out.s'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = wait_timeout(proc, 120)

            error_comp = False

            if result != None:

                try:
                    for line in iter(proc.stderr.readline, ''):
                        print line
                        if check_error(line):
                            error_comp = True
                            break
                    for line in iter(proc.stdout.readline, ''):
                        print line
                        if check_error(line):
                            error_comp = True
                            break
                except:
                    error_comp = True

            else:
                error_comp = True

            if error_comp:
                errors += 1
                continue

            #print 'comp succesful'

            proc = subprocess.Popen(['./iaca','-arch',args.cpu,'-reduceout','test.o'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            start_time = time.time()
            result = wait_timeout(proc, 10)
            end_time = time.time()

            if result != None:

                error_lines = False
                for line in iter(proc.stderr.readline, ''):
                    print line
                    if check_error(line):
                        error_lines = True
                        break

                if error_lines == False:
                    success += 1
                    for line in iter(proc.stdout.readline, ''):
                        found = re.search('Block Throughput: ([0-9]+\.?[0-9]*) Cycles.*',line)
                        if found:
                            #print found.group(0)
                            cycles = float(found.group(1))
                            if cycles != 0:
                                total_time += end_time - start_time
                                print cycles
                                if not args.tp:
                                    insert_time_value(cnx, row[1], cycles, args.arch)
                            break
                else:
                    for line in final_bb:
                        print line[:-1]
                    errors += 1

            else:
                print 'error not completed'
                not_finished += 1

        print total, success, errors, not_finished, except_errors, total_time

    cnx.close()
