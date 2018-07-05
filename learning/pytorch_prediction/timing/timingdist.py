from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import common.utilities as ut
from tqdm import tqdm


def example_plot():

    x = np.random.choice(10,10)
    y = np.random.choice(10,10)
    plt.plot(x,y)
    plt.show()

class CodeTimes:
    
    def __init__(self,code_id, code, program, rel_addr, sym_dict, mem_offset):
        self.code_id = code_id
        self.code = []
        for token in code.split(','):
            if token != '':
                self.code.append(ut.get_name(int(token),sym_dict,mem_offset))
        self.program = program
        self.rel_addr = rel_addr
        self.percentage = 10
        self.times = []

    def add_times(self,times):

        self.min_time_threshold = 44
        self.min_count_threshold = 1

        self.count = 0

        self.mode_time = 0
        self.mode_count = 0
                
        self.min_time = 2e8
        self.min_count = 0

        for row in times:
            if row[0] <= self.min_time_threshold: #if the times are too low
                continue
            inserted = False
            try:
                for i,time in enumerate(self.times):
                    if (abs(row[0] - time[0]) * 100) / time[0] < self.percentage:
                        new_time = (time[0] * time[1] + row[0] * row[1]) / (time[1] + row[1])
                        new_count = time[1] + row[1]
                        self.times[i] = [new_time, new_count]
                        inserted = True
                        break
                if not inserted:
                    self.times.append([row[0],row[1]])
            except:
                print 'error' 
                print self.times

        for time in self.times:
            self.count += time[1]
  
        if self.count >= self.min_count_threshold:
            for time in self.times:
                if time[1] >= self.mode_count:
                    self.mode_time = time[0]
                    self.mode_count = time[1]
                if time[0] < self.min_time:
                    self.min_time = time[0]
                    self.min_count = time[1]
                    
            self.mode_count = self.mode_count * 100.0 / self.count
            self.min_count = self.min_count * 100.0 / self.count
        

    def print_times(self):
        if self.count > self.min_count_threshold:
            print self.times
            print self.count
            print self.mode_time, self.mode_count
            print self.min_time, self.min_count



    def print_code(self):
        if self.count > self.min_count_threshold:
            print self.program, self.rel_addr
            print self.code
        


if __name__ == '__main__':

    cnx = ut.create_connection('timing0518')
    
    sql = 'SELECT code_id, code, program, rel_addr from code'
    rows = ut.execute_query(cnx, sql, True)

    offsets_file = '../inputs/offsets.txt'
    encoding_file = '../inputs/encoding.h'
    sym_dict, mem_offset = ut.get_sym_dict(offsets_file, encoding_file)
    
    codes = []
    for row in tqdm(rows):
        code = CodeTimes(row[0],row[1],row[2],row[3],sym_dict,mem_offset)
        sql = 'SELECT time, count FROM times WHERE code_id = ' + str(row[0])
        times = ut.execute_query(cnx, sql, True)
        code.add_times(times)
        codes.append(code)

    #for code in codes:
    #    code.print_code()
    #    code.print_times()


    percentage = []
    over = 0.0
    for code in codes:
        if code.count >= code.min_count_threshold:
            percentage.append(code.mode_count)
        if code.mode_count > 20:
            over += 1

    print 'total ' + str(len(codes))
    print 'total above count threshold ' + str(len(percentage))
    print 'total over ' + str(over)
    print 'percentage ' + str(float(over) / float(len(codes)))

    amount = range(len(percentage))
    plt.plot(amount, percentage)
    plt.show()

    cnx.close()
