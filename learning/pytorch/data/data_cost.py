import numpy as np
import random
import word2vec.word2vec as w2v
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
from tqdm import tqdm
from data import Data
import matplotlib.pyplot as plt
import statistics

import sys
sys.path.append('..')

import common_libs.utilities as ut


class DataItem:

    def __init__(self, x, y, block):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = None


class DataCost(Data):

    def __init__(self, data=None):
        super(DataCost, self).__init__(data)
        self.time_percentage = 10
        self.threshold = 30

        self.min_count_threshold = 1
        self.min_time_threshold = 44
        self.max_time_threshold = 10000


    def get_time_mode(self, times):

        self.mode_count = 0
        self.mode_time = 0

        self.times = []
        for row in times:
            if row[0] <= self.min_time_threshold or row[0] > self.max_time_threshold:
                continue
            inserted = False
            try:
                for i,time in enumerate(self.times):
                    if (abs(row[0] - time[0]) * 100) / time[0] < self.time_percentage:
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

        self.count = 0
        for time in self.times:
            self.count += time[1]

        if self.count >= self.min_count_threshold:
            for time in self.times:
                if time[1] >= self.mode_count:
                    self.mode_time = time[0]
                    self.mode_count = time[1]

            self.mode_count = self.mode_count * 100.0 / self.count

        if self.count > 0 and (self.mode_count >= self.threshold):
            return self.mode_time, self.mode_count
        else:
            return None, None

    def update_times(self, cnx):

        sql = 'SELECT code_id FROM code'
        rows = ut.execute_query(cnx, sql, True)

        row_count = 0
        for row in tqdm(rows):
            sql = 'UPDATE code SET time=NULL WHERE code_id=' + str(row[0])
            ut.execute_query(cnx, sql, False)
        cnx.commit()

        mode_dist = dict()
        for row in tqdm(rows):
            sql = 'SELECT time, count FROM times WHERE code_id = ' + str(row[0])
            times = ut.execute_query(cnx, sql, True)
            mode, count = self.get_time_mode(times)

            if mode != None:
                if mode in mode_dist:
                    mode_dist[mode] += 1
                else:
                    mode_dist[mode] = 1

                row_count += 1
                sql = 'UPDATE code SET time=' + str(mode) + ' WHERE code_id=' + str(row[0])
                ut.execute_query(cnx, sql, False)
                sql = 'SELECT time from code WHERE code_id=' + str(row[0])
                res = ut.execute_query(cnx, sql, True)
                assert res[0][0] == mode

        print len(rows)
        print row_count
        print mode_dist

        cnx.commit()

    def print_times(self, f, text, time):

        for t in text:
             f.write('%s,' % t)
        f.write(' %d\n' % time)

    def clean_data(self):
        #remove skew of data
        temp_data = []
        per_region_count = dict()
        max_per_region = 4000
        region_size = 10

        self.max_time = min(1000, self.max_time)

        omitted = 0
        for item in self.data:
            y = item.y
            if y > self.max_time:
                omitted += 1
                continue

            region = y // region_size
            if region in per_region_count:
                per_region_count[region] += 1
            else:
                per_region_count[region] = 1

            if per_region_count[region] <= max_per_region:
                temp_data.append(item)

        print 'omitted ' + str(omitted)

        self.data = temp_data

        print 'after removing skew...'
        print len(self.data)

        #randomize
        shuffled = range(len(self.data))
        random.shuffle(shuffled)

        temp_data = []
        for i in shuffled:
            temp_data.append(self.data[i])

        self.data = temp_data

        self.plot_histogram(self.data)


    def count_cost_dist(self,data):
        costs = dict()
        for item in data:
            if item.y in costs:
                costs[item.y] += 1
            else:
                costs[item.y] = 1

        for key in sorted(costs.iterkeys()):
            sys.stdout.write("%d: %d, " % (key, costs[key]))
        sys.stdout.write('\n')


    def prepare_for_classification(self):

        for item in self.data:
            one_hot = [0] * self.max_time
            one_hot[item.y - 1] = 1
            item.y = one_hot

        return self.max_time


    def get_timing_data(self, cnx, arch):

        data = []

        #assumes code_token and code_id
        for row in tqdm(self.raw_data):

            if len(row[0]) == 0:
                continue

            code_id = row[1]

            sql = 'SELECT kind, time from times where code_id=' + str(code_id) + ' and arch=' + str(arch)

            values = []
            times = ut.execute_query(cnx,sql, True)
            for time in times:
                if time[0] == 'actual':
                    values.append(time[1])

            if len(values) == 0:
                continue

            final_value = statistics.mean(values)

            data.append((row[0],final_value,row[2],row[1]))


        self.raw_data = data

        print 'timing values registered for ' + str(len(data)) + ' items'


class DataTokenEmbedding(DataCost):

    def __init__(self, data=None):
        super(DataTokenEmbedding, self).__init__(data)

    def prepare_data(self):

        self.data = []
        times = dict()

        for row in tqdm(self.raw_data):
            if len(row[0]) > 0 and row[1] != None:
                code = []
                for token in row[0]:
                    code.append(self.word2id.get(token,0))

                mode = row[1]

                if mode <= 20 or mode > 10000:
                    continue

                if mode in times:
                    times[mode] += 1
                else:
                    times[mode] = 1

                item = DataItem(code, mode, None)

                if len(row) > 3:
                    item.code_id = row[3]

                self.data.append(item)

        self.max_time = max(times)
        print len(self.raw_data), len(self.data)




class DataInstructionEmbedding(DataCost):

    def __init__(self, data=None):
        super(DataInstructionEmbedding, self).__init__(data)


    def prepare_data(self):

        self.data = []
        times = dict()

        for row in tqdm(self.raw_data):
            if len(row[0]) > 0 and row[1] != None:
                code = []
                ins = []
                for token in row[0]:
                    if token >= self.opcode_start and token < self.mem_start:
                        if len(ins) != 0:
                            code.append(ins)
                            ins = []
                    ins.append(self.word2id.get(token,0))
                if len(ins) != 0:
                    code.append(ins)
                    ins = []


                mode = row[1]

                if mode <= 20 or mode > 10000:
                    continue

                block = ut.create_basicblock(row[0])
                block.create_dependencies()


                if mode in times:
                    times[mode] += 1
                else:
                    times[mode] = 1

                item = DataItem(code, mode, block)

                if len(row) > 3:
                    item.code_id = row[3]

                self.data.append(item)

        self.max_time = max(times)
        print len(self.raw_data), len(self.data)
