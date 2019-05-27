import collections
import mysql.connector
import struct
import sys
from mysql.connector import errorcode
import random
import re
import os
import tempfile
from typing import Dict, FrozenSet, Optional, Tuple, Union

#mysql specific functions
def create_connection(database=None, user=None, password=None, port=None):
    args = {}

    option_files = list(filter(os.path.exists, map(os.path.abspath, map(os.path.expanduser, [
        '/etc/my.cnf',
        '~/.my.cnf',
    ]))))

    if option_files:
        args['option_files'] = option_files
    if database:
        args['database'] = database
    if user:
        args['user'] = user
    if password:
        args['password'] = password
    if port:
        args['port'] = port

    cnx = None
    try:
        cnx = mysql.connector.connect(**args)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

    return cnx

def get_mysql_config(filename):

    config = dict()
    with open(filename,'r') as f:
        for line in f:
            found = re.search('([a-zA-Z\-]+) *= *\"*([a-zA-Z0-9#\./]+)\"*', line)
            if found:
                config[found.group(1)] = found.group(2)
    return config


def create_connection_from_config(config_file, database=None):

    config = get_mysql_config(config_file)
    cnx = create_connection(user=config['user'],password=config['password'],port=config['port'],database=database)
    return cnx

def execute_many(cnx, sql, values):
    cur = cnx.cursor(buffered=True)
    cur.executemany(sql, values)


def execute_query(cnx, sql, fetch, multi=False):
    cur = cnx.cursor(buffered=True)
    cur.execute(sql,multi)
    if fetch:
        return cur.fetchall()
    else:
        return None

#data reading function
def get_data(cnx, format, cols, limit=None):
    try:
        cur = cnx.cursor(buffered=True)

        #code column is mandatory
        columns = 'code_token'
        for col in cols:
            columns += ',' + col
        columns += ''

        sql = 'SELECT ' + columns + ' FROM code'
        if limit is not None:
            sql += ' LIMIT {}'.format(limit)

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


#dynamorio specific encoding details - tokenizing
def get_opcode_opnd_dict(opcode_start, opnd_start):
    sym_dict = dict()

    filename = os.environ['ITHEMAL_HOME'] + '/common/inputs/encoding.h'

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

def read_offsets():
    offsets_filename = os.environ['ITHEMAL_HOME'] + '/common/inputs/offsets.txt'
    offsets = list()
    with open(offsets_filename,'r') as f:
        for line in f:
            for value in line.split(','):
                offsets.append(int(value))
        f.close()
    assert len(offsets) == 5
    return offsets

def get_sym_dict():
    # type: Tuple[Dict[int, str], int]

    offsets = read_offsets()
    sym_dict = get_opcode_opnd_dict(opcode_start = offsets[0],opnd_start = offsets[1])

    sym_dict[offsets[2]] = 'int_immed'
    sym_dict[offsets[3]] = 'float_immed'

    return sym_dict, offsets[4]

_REGISTER_ALIASES = (
    {'REG_RAX', 'REG_EAX', 'REG_AX', 'REG_AH', 'REG_AL'},
    {'REG_RBX', 'REG_EBX', 'REG_BX', 'REG_BH', 'REG_BL'},
    {'REG_RCX', 'REG_ECX', 'REG_CX', 'REG_CH', 'REG_CL'},
    {'REG_RDX', 'REG_EDX', 'REG_DX', 'REG_DH', 'REG_DL'},
    {'REG_RSP', 'REG_ESP', 'REG_SP'},
    {'REG_RBP', 'REG_EBP', 'REG_BP'},
    {'REG_RSI', 'REG_ESI', 'REG_SI'},
    {'REG_RDI', 'REG_EDI', 'REG_DI'},
    {'REG_R8', 'REG_R8D', 'REG_R8W', 'REG_8L'},
    {'REG_R9', 'REG_R9D', 'REG_R9W', 'REG_9L'},
    {'REG_R10', 'REG_R10D', 'REG_R10W', 'REG_10L'},
    {'REG_R11', 'REG_R11D', 'REG_R11W', 'REG_11L'},
    {'REG_R12', 'REG_R12D', 'REG_R12W', 'REG_12L'},
    {'REG_R13', 'REG_R13D', 'REG_R13W', 'REG_13L'},
    {'REG_R14', 'REG_R14D', 'REG_R14W', 'REG_14L'},
    {'REG_R15', 'REG_R15D', 'REG_R15W', 'REG_15L'},
)
_REGISTER_ALIAS_MAP = {reg: regset for regset in _REGISTER_ALIASES for reg in regset}
def _get_canonical_operand(op):
    return _REGISTER_ALIAS_MAP.get(_global_sym_dict.get(op, None), op)

_REGISTER_CLASSES = tuple(map(frozenset, (
    {'REG_RAX', 'REG_RCX', 'REG_RDX', 'REG_RBX', 'REG_RSP', 'REG_RBP', 'REG_RSI',
     'REG_RDI', 'REG_R8', 'REG_R9', 'REG_R10', 'REG_R11', 'REG_R12', 'REG_R13',
     'REG_R14', 'REG_R15'},
    {'REG_EAX', 'REG_ECX', 'REG_EDX', 'REG_EBX', 'REG_ESP', 'REG_EBP', 'REG_ESI',
     'REG_EDI', 'REG_R8D', 'REG_R9D', 'REG_R10D', 'REG_R11D', 'REG_R12D', 'REG_R13D',
     'REG_R14D', 'REG_R15D'},
    {'REG_AX', 'REG_CX', 'REG_DX', 'REG_BX', 'REG_SP', 'REG_BP', 'REG_SI',
     'REG_DI', 'REG_R8W', 'REG_R9W', 'REG_R10W', 'REG_R11W', 'REG_R12W', 'REG_R13W',
     'REG_R14W', 'REG_R15W'},
    {'REG_AL', 'REG_CL', 'REG_DL', 'REG_BL', 'REG_AH', 'REG_CH', 'REG_DH',
     'REG_BH', 'REG_R8L', 'REG_R9L', 'REG_R10L', 'REG_R11L', 'REG_R12L', 'REG_R13L',
     'REG_R14L', 'REG_R15L'},
)))

def get_register_class(reg):
    # type: Union[str, int] -> Optional[FrozenSet[str]]

    if isinstance(reg, int):
        reg = _global_sym_dict.get(reg)

    for cls in _REGISTER_CLASSES:
        if reg in cls:
            return cls

    return None

def get_name(val,sym_dict,mem_offset):
    if val >= mem_offset:
        return 'mem_' + str(val - mem_offset)
    elif val < 0:
        return 'delim'
    else:
        return sym_dict[val]

def get_percentage_error(predicted, actual):

    errors = []
    for pitem, aitem in zip(predicted, actual):

        if type(pitem) == list:
            pitem = pitem[-1]
            aitem = aitem[-1]

        error = abs(float(pitem) - float(aitem)) * 100.0 / float(aitem)

        errors.append(error)

    return errors

_global_sym_dict, _global_mem_start = get_sym_dict()
_global_sym_dict_rev = {v:k for (k, v) in _global_sym_dict.items()}

#calculating static properties of instructions and basic blocks
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

    def clone(self):
        return Instruction(self.opcode, self.srcs[:], self.dsts[:], self.num)

    def print_instr(self):
        print self.num, self.opcode, self.srcs, self.dsts
        num_parents = [parent.num for parent in self.parents]
        num_children = [child.num for child in self.children]
        print num_parents, num_children

    def __str__(self):
        return self.intel

    def has_mem(self):
        return any(operand >= _global_mem_start for operand in self.srcs + self.dsts)

    def is_idempotent(self):
        return len(set(self.srcs) & set(self.dsts)) == 0

class InstructionReplacer(object):
    def __init__(self, regexp_intel, replacement_intel,
                 replacement_srcs, replacement_dsts):
        self.regexp_intel = re.compile(regexp_intel)
        self.replacement_intel = replacement_intel
        self.replacement_srcs = replacement_srcs
        self.replacement_dsts = replacement_dsts

    def replace(self, instr, unused_registers):
        if instr.has_mem():
            return None

        match = self.regexp_intel.match(instr.intel)
        if match is None:
            return None

        unused_set = None
        for operand in instr.dsts:
            op_cls = get_register_class(operand)
            if op_cls is None:
                continue

            m_unused_set = op_cls & unused_registers
            if unused_set is not None:
                assert unused_set == m_unused_set, 'Did not expect mix of operand types'

            unused_set = m_unused_set

        if not unused_set:
            return None

        unused = list(unused_set)
        unused_intel = list(map(lambda x: x[x.rindex('_')+1:].lower(), unused))
        unused_token = list(map(_global_sym_dict_rev.get, unused))

        new_instr = instr.clone()
        new_instr.intel = self.replacement_intel.format(
            unused=unused_intel,
            **match.groupdict()
        )

        new_instr.srcs = list(map(int, map(lambda x: x.format(
            srcs=instr.srcs,
            dsts=instr.dsts,
            unused=unused_token,
        ), self.replacement_srcs)))

        new_instr.dsts = list(map(int, map(lambda x: x.format(
            srcs=instr.srcs,
            dsts=instr.dsts,
            unused=unused_token,
        ), self.replacement_dsts)))

        return new_instr

def _two_way_replacer(opcode):
    return InstructionReplacer(
        r'{}\s+(?P<op1>\w+),\s+(?P<op2>\w+)'.format(opcode),
        r'{} {{unused[0]}}, {{op2}}'.format(opcode),
        ['{srcs[0]}', '{unused[0]}'],
        ['{unused[0]}'],
    )

def _three_way_replacer(opcode):
    return InstructionReplacer(
        r'{}\s+(?P<op1>\w+),\s+(?P<op2>\w+),\s+(?P<op3>\w+)'.format(opcode),
        r'{} {{unused[0]}}, {{op2}}, {{op3}}'.format(opcode),
        ['{srcs[0]}', '{srcs[1]}'],
        ['{unused[0]}'],
    )

replacers = (
    _two_way_replacer('add'), _two_way_replacer('sub'), _two_way_replacer('and'),
    _two_way_replacer('or'), _two_way_replacer('xor'), _two_way_replacer('shl'),
    _two_way_replacer('shr'), _two_way_replacer('sar'),
    _three_way_replacer('imul'),
)


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
        for dst in map(_get_canonical_operand, instr.dsts):
            for i in range(n + 1, len(self.instrs), 1):
                dst_instr = self.instrs[i]
                if dst in map(_get_canonical_operand, dst_instr.srcs):
                    if not dst_instr in instr.children:
                        instr.children.append(dst_instr)
                if dst in map(_get_canonical_operand, dst_instr.dsts): #value becomes dead here
                    break

    def find_defs(self, n):

        instr = self.instrs[n]
        for src in map(_get_canonical_operand, instr.srcs):
            for i in range(n - 1, -1, -1):
                src_instr = self.instrs[i]
                if src in map(_get_canonical_operand, src_instr.dsts):
                    if not src_instr in instr.parents:
                        instr.parents.append(src_instr)
                    break

    def create_dependencies(self):

        for n in range(len(self.instrs)):
            self.find_defs(n)
            self.find_uses(n)

    def get_dfs(self):
        dfs = collections.defaultdict(set)

        for instr in self.instrs[::-1]:
            frontier = {instr}
            while frontier:
                n = frontier.pop()
                if n in dfs:
                    dfs[instr] |= dfs[n]
                    continue

                for c in n.children:
                    if c in dfs[instr] or c in frontier:
                        continue
                    frontier.add(c)
                dfs[instr].add(n)

        return dfs

    def transitive_closure(self):
        dfs = self.get_dfs()
        for instr in self.instrs:
            transitive_children = set(n for c in instr.children for n in dfs[c])
            instr.children = list(transitive_children)
            for child in instr.children:
                if instr not in child.parents:
                    child.parents.append(instr)

    def transitive_reduction(self):
        dfs = self.get_dfs()
        for instr in self.instrs:

            transitively_reachable_children = set()
            for child in instr.children:
                transitively_reachable_children |= dfs[child] - {child}

            for child in transitively_reachable_children:
                if child in instr.children:
                    instr.children.remove(child)
                    child.parents.remove(instr)

    def random_forward_edges(self, frequency):
        '''Add forward-facing edges at random to the instruction graph.

        There are n^2/2 -1 considered edges (where n is the number of
        instructions), so to add 5 edges in expectation, one would
        provide frequency=5/(n^2/2-1)

        '''
        n_edges_added = 0
        for head_idx, head_instr in enumerate(self.instrs[:-1]):
            for tail_instr in self.instrs[head_idx+1:]:
                if random.random() < frequency:
                    if tail_instr not in head_instr.children:
                        head_instr.children.append(tail_instr)
                        tail_instr.parents.append(head_instr)
                        n_edges_added += 1

        return n_edges_added

    def remove_edges(self):
        for instr in self.instrs:
            instr.parents = []
            instr.children = []

    def linearize_edges(self):
        for fst, snd in zip(self.instrs, self.instrs[1:]):
            if snd not in fst.children:
                fst.children.append(snd)
            if fst not in snd.parents:
                snd.parents.append(fst)

    def find_roots(self):
        roots = []
        for instr in self.instrs:
            if len(instr.parents) == 0:
                roots.append(instr)
        return roots

    def find_leaves(self):
        leaves = []
        for instr in self.instrs:
            if len(instr.children) == 0:
                leaves.append(instr)

        return leaves

    def gen_reorderings(self, single_perm=False):
        self.create_dependencies()

        def _gen_reorderings(prefix, schedulable_instructions, mem_q):
            mem_q = mem_q[:]
            has_pending_mem = any(instr.has_mem() for instr in schedulable_instructions)
            has_activated_mem = mem_q and all(parent in prefix for parent in mem_q[0].parents)

            if has_activated_mem and not has_pending_mem:
                schedulable_instructions.append(mem_q.pop(0))

            if len(schedulable_instructions) == 0:
                return [prefix]

            reorderings = []
            def process_index(i):
                instr = schedulable_instructions[i]
                # pop this instruction
                rest_scheduleable_instructions = schedulable_instructions[:i] + schedulable_instructions[i+1:]
                rest_prefix = prefix + [instr]

                # add all activated children
                for child in instr.children:
                    if all(parent in rest_prefix for parent in child.parents):
                        if not child.has_mem():
                            rest_scheduleable_instructions.append(child)

                reorderings.extend(_gen_reorderings(rest_prefix, rest_scheduleable_instructions, mem_q))

            if single_perm:
                process_index(random.randrange(len(schedulable_instructions)))
            else:
                for i in range(len(schedulable_instructions)):
                    process_index(i)

            return reorderings

        return _gen_reorderings(
            [],
            [i for i in self.find_roots() if not i.has_mem()],
            [i for i in self.instrs if i.has_mem()],
        )

    def sample_reordering(self):
        # TODO: THIS VIOLATES FALSE DEPENDENCIES
        prefix = []
        enabled = []
        enabled_mem = []
        some_mem_enabled = False

        def is_enabled(i):
            return all(p in prefix for p in instr.parents)

        for instr in self.instrs:
            if len(instr.parents) == 0:
                if instr.has_mem():
                    if not some_mem_enabled:
                        enabled.append(instr)
                        some_mem_enabled = True
                    else:
                        enabled_mem.append(instr)
                else:
                    enabled.append(instr)

        while enabled or enabled_mem:
            to_schedule = random.randrange(len(enabled))
            instr = enabled.pop(to_schedule)
            prefix.append(instr)
            if instr.has_mem():
                if enabled_mem:
                    enabled.append(enabled_mem.pop(0))
                else:
                    some_mem_enabled = False
            for ch in instr.children:
                if is_enabled(ch):
                    if ch.has_mem():
                        if some_mem_enabled:
                            enabled_mem.append(ch)
                        else:
                            enabled.append(ch)
                            some_mem_enabled = True
                    else:
                        enabled.append(ch)

        return prefix

    def paths_of_block(self):
        # type: () -> List[List[ut.Instruction]]
        def paths_of_instr(i, parents):
            # type: (ut.Instruction, List[ut.Instruction]) -> List[List[ut.Instruction]]
            new_parents = parents + [i]
            if i.children:
                return sum((paths_of_instr(c, new_parents) for c in i.children), [])
            else:
                return [new_parents]

        return sum((paths_of_instr(i, []) for i in self.find_roots()), [])

    def draw(self, to_file=False, file_name=None, view=True):
        if to_file and not file_name:
            file_name = tempfile.NamedTemporaryFile(suffix='.gv').name

        from graphviz import Digraph

        dot = Digraph()
        for instr in self.instrs:
            dot.node(str(id(instr)), str(instr))
            for child in instr.children:
                dot.edge(str(id(instr)), str(id(child)))

        if to_file:
            dot.render(file_name, view=view)
            return dot, file_name
        else:
            return dot

    def has_mem(self):
        return any(map(Instruction.has_mem, self.instrs))

    def has_no_dependencies(self):
        return all(len(i.parents) == 0 and len(i.children) == 0 for i in self.instrs)

    def has_linear_dependencies(self):
        if len(self.instrs) <= 1:
            return True

        return (
            len(self.instrs[0].children) == 1 and
            all(len(i.parents) == 1 and len(i.children) == 1 for i in self.instrs[1:-1]) and
            len(self.instrs[-1].parents) == 1
        )

def generate_duplicates(instrs, max_n_dups):
    for idx in range(len(instrs) - 1, -1, -1):
        instr = instrs[idx]
        unused_regs = unused_registers_at_point(instrs, idx)
        for replacer in replacers:
            res = replacer.replace(instr, unused_regs)
            if res is None:
                continue

            augmentations = []
            new_instrs = instrs[:]
            for i in range(max_n_dups):
                unused_regs = unused_registers_at_point(new_instrs, idx)
                aug_instr = replacer.replace(instr, unused_regs)
                if not aug_instr:
                    break
                new_instrs.insert(idx, aug_instr)
                augmentations.append(new_instrs[:])

            return augmentations

    return []


def unused_registers_at_point(instrs, idx):
    if idx < 0 or idx > len(instrs):
        raise ValueError('{} is not a valid index'.format(idx))

    unused_regs = set()
    for cls in _REGISTER_CLASSES:
        unused_regs |= cls
    for instr in instrs[idx:]:
        for src in instr.srcs:
            unused_regs -= {_global_sym_dict.get(src)}
        for dst in instr.dsts:
            unused_regs -= {_global_sym_dict.get(dst)}

    return unused_regs


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


if __name__ == "__main__":
    cnx = create_connection()
    cur = cnx.cursor(buffered = True)

    sql = 'SELECT code_id, code_token from  code where program = \'2mm\' and rel_addr = 4136'

    cur.execute(sql)

    rows = cur.fetchall()

    sym_dict, mem_start = get_sym_dict()

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
