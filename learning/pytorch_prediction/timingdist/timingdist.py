from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import common.utilities as ut


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
        for row in times:
            if row[0] < 1:
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
                print self.times

    def print_times(self):
        print self.times
        count = 0
        for time in self.times:
            count += time[1]
        self.count = 0
        self.av_time = 0
        try:
            for time in self.times:
                if time[1] >= self.count:
                    self.av_time = time[0]
                    self.count = time[1]
            self.count = self.count * 100 / count
            print self.av_time, self.count
        except:
            self.count = 100
            print 0, self.count
            



    def print_code(self):
        print self.program, self.rel_addr
        print self.code
        


if __name__ == '__main__':

    cnx = ut.create_connection('timingmodel')
    
    cur = cnx.cursor(buffered=True)
    sql = 'SELECT code_id, code, program, rel_addr from code'
    cur.execute(sql)

    offsets_file = '../inputs/offsets.txt'
    encoding_file = '../inputs/encoding.h'
    sym_dict, mem_offset = ut.get_sym_dict(offsets_file, encoding_file)
    rows = cur.fetchall()

    codes = []
    print len(rows)
    for i,row in enumerate(rows):
        new_code = CodeTimes(row[0],row[1],row[2],row[3],sym_dict,mem_offset)
        sql = 'SELECT time, count FROM times WHERE code_id = ' + str(row[0])
        cur.execute(sql)
        times = cur.fetchall()
        new_code.add_times(times)
        codes.append(new_code)
        if i % 10000 == 0:
            print i

    for code in codes:
        code.print_code()
        code.print_times()


    amount = range(len(codes))
    percentage = []
    over = 0
    for code in codes:
        percentage.append(code.count)
        if code.count > 70:
            over += 1

    print len(codes), over, over / len(codes)

    plt.plot(amount, percentage)
    plt.show()

    cnx.close()
