import mysql.connector
import struct
import sys
from mysql.connector import errorcode
import re
import os


#mysql specific functions
def create_connection(database, user, password, port):
    cnx = None
    try:
        cnx = mysql.connector.connect(user=user,password=password,database=database,port=port);
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    return cnx

def execute_many(cnx, sql, values):
    cur = cnx.cursor(buffered=True)
    cur.executemany(sql, values)


def execute_query(cnx, sql, fetch):
    cur = cnx.cursor(buffered=True)
    cur.execute(sql)
    # if result.with_rows:
    #     print("Rows produced by statement '{}':".format(
    #         result.statement))
    # else:
    #     print("Number of rows affected by statement '{}': {}".format(
    #         result.statement, result.rowcount))
    if fetch:
        return cur.fetchall()
    else:
        return None


#dynamorio specific encoding details
def get_opcode_opnd_dict(opcode_start, opnd_start, filename):
    sym_dict = dict()
    with open(filename,'r') as f:
        opcode_num = opcode_start
        opnd_num = opnd_start
        for line in f:
            opcode_re = re.search('/\*.*\*/.*OP_([a-zA-Z_0-9]+),.*', line)
            if opcode_re != None:
                sym_dict[opcode_num] = opcode_re.group(1)
                opcode_num = opcode_num + 1
            opnd_re = re.search('.*DR_([A-Za-z_0-9]+),.*', line)
            if opnd_re != None:
                sym_dict[opnd_num] = opnd_re.group(1)
                opnd_num = opnd_num + 1
        f.close()

    return sym_dict

def read_offsets(filename):
    offsets_filename = filename
    offsets = list()
    with open(offsets_filename,'r') as f:
        for line in f:
            for value in line.split(','):
                offsets.append(int(value))
        f.close()
    assert len(offsets) == 5
    return offsets
    
def get_sym_dict(offsets_filename,encoding_filename):

    offsets = read_offsets(offsets_filename)
    sym_dict = get_opcode_opnd_dict(opcode_start = offsets[0],opnd_start = offsets[1], filename = encoding_filename)
   
    sym_dict[offsets[2]] = 'int_immed'
    sym_dict[offsets[3]] = 'float_immed'

    return sym_dict, offsets[4]

def get_name(val,sym_dict,mem_offset):
    if val >= mem_offset:
        return 'mem_' + str(val - mem_offset)
    elif val < 0:
        return 'delim'
    else:
        return sym_dict[val]

 
#data reading function
def get_data(cnx, format, cols):
    try:
        cur = cnx.cursor(buffered=True)

        #code column is mandatory
        columns = 'code_token'
        for col in cols:
            columns += ',' + col
        columns += ''

        sql = 'SELECT ' + columns + ' FROM code'
        print sql
        data = list()
        cur.execute(sql)
        print cur.rowcount
        row = cur.fetchone()
        while row != None:
            item = list()
            code = list()
            if format == 'text':
                for value in row[0].split(','):
                    if value != '':
                        code.append(int(value))
            elif format == 'bin':
                if len(row[0]) % 2 != 0:
                    row = cur.fetchone()
                    continue
                for i in range(0,len(row[0]),2): 
                    slice = row[0][i:i+2]
                    convert = struct.unpack('h',slice)
                    code.append(int(convert[0]))
            
            item.append(code)
            for i in range(len(cols)):
                item.append(row[i + 1])
            data.append(item)
            row = cur.fetchone()
    except Exception as e:
        print e
    else:
        return data

def get_percentage_error(predicted, actual):

    errors = []
    for pitem, aitem in zip(predicted, actual):
        
        if type(pitem) == list:
            pitem = pitem[-1]
            aitem = aitem[-1]
        
        error = abs(float(pitem) - float(aitem)) * 100.0 / float(aitem)

        errors.append(error)

    return errors
        

#calculating static properties of instructions and basic blocks

def create_basicblock(tokens):

    opcode = None
    srcs = []
    dsts = []
    mode = 0

    mode = 0
    instrs = []
    for item in tokens:
        if item == -1:
            mode += 1
            if mode > 2:
                mode = 0
                instr = Instruction(opcode,srcs,dsts,len(instrs))
                instrs.append(instr)
                opcode = None
                srcs = []
                dsts = []
                continue
        else:
            if mode == 0:
                opcode = item
            elif mode == 1:
                srcs.append(item)
            else:
                dsts.append(item)

    block = BasicBlock(instrs)
    return block


class Instruction:
    
    def __init__(self, opcode, srcs, dsts, num):
        self.opcode = opcode
        self.num = num
        self.srcs = srcs
        self.dsts = dsts
        self.parents = []
        self.children = []

        #for lstms
        self.lstm = None
        self.hidden = None
        self.tokens = None


    def print_instr(self):
        print self.num, self.opcode, self.srcs, self.dsts
        num_parents = [parent.num for parent in self.parents]
        num_children = [child.num for child in self.children]
        print num_parents, num_children

class BasicBlock:

    def __init__(self, instrs):
        self.instrs = instrs
        self.span_values = [0] * len(self.instrs)

    def num_instrs(self):
        return len(self.instrs)

    def num_span(self, instr_cost):
        
        for i in range(len(self.instrs)):
            self.span_rec(i, instr_cost)

        if len(self.instrs) > 0:
            return max(self.span_values)
        else:
            return 0

    def print_block(self):
        for instr in self.instrs:
            instr.print_instr()


    def span_rec(self, n, instr_cost):

        if self.span_values[n] != 0:
            return self.span_values[n]

        src_instr = self.instrs[n]
        span = 0
        dsts = []
        for dst in src_instr.dsts:
            dsts.append(dst)

        for i in range(n + 1, len(self.instrs)):
            dst_instr = self.instrs[i]
            for dst in dsts:
                found = False
                for src in dst_instr.srcs:
                    if(dst == src):
                        ret = self.span_rec(i, instr_cost)
                        if span < ret:
                            span = ret
                        found = True
                        break
                if found:
                    break
            dsts = list(set(dsts) - set(dst_instr.dsts)) #remove dead destinations
        
        if src_instr.opcode in instr_cost:
            cost = instr_cost[src_instr.opcode]
        else:
            src_instr.print_instr()
            cost = 1

        #assert cost == 1

        self.span_values[n] = span + cost
        return self.span_values[n]
                        

    def find_uses(self, n):

        instr = self.instrs[n]
        for dst in instr.dsts:
            for i in range(n + 1, len(self.instrs), 1):
                dst_instr = self.instrs[i]
                if dst in dst_instr.srcs:
                    if not dst_instr in instr.children:
                        instr.children.append(dst_instr)
                if dst in dst_instr.dsts: #value becomes dead here
                    break

    def find_defs(self, n):

        instr = self.instrs[n]
        for src in instr.srcs:
            for i in range(n - 1, -1, -1):
                src_instr = self.instrs[i]
                if src in src_instr.dsts:
                    if not src_instr in instr.parents:
                        instr.parents.append(src_instr)
                    break
        
    def create_dependencies(self):

        for n in range(len(self.instrs)):
            self.find_defs(n)
            self.find_uses(n)


    def find_roots(self):
        roots = []
        for instr in self.instrs:
            if len(instr.children) == 0:
                roots.append(instr)

        return roots

        

if __name__ == "__main__":
    
    cnx = create_connection('costmodel0404')
    cur = cnx.cursor(buffered = True)
    
    sql = 'SELECT code_id, code from  code where program = \'2mm\' and rel_addr = 4136'

    cur.execute(sql)

    rows = cur.fetchall()

    sym_dict, mem_start = get_sym_dict('/data/scratch/charithm/projects/cmodel/database/offsets.txt')

    for row in rows:
        print row[0]
        code = []
        for val in row[1].split(','):
            if val != '':
                code.append(get_name(int(val),sym_dict,mem_start))
        print code


    sql = 'SELECT time from times where code_id = ' + str(rows[0][0])
    cur.execute(sql)
    rows = cur.fetchall()

    times = [int(t[0]) for t in rows]
    print sorted(times)
        
