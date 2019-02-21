import gdb

import re
from collections import namedtuple

# where and "how much" data is loaded or stored
MemAccess = namedtuple('MemAccess', ['loc', 'size'])

BEGIN_LABEL = 'test_begin'
END_LABEL = 'test_end'

OUT_FILE = 'accesses.txt'

mem_re = re.compile(r'.*\[(.*)\].*')
# approximation of a register inside a memory access
reg_re = re.compile(r'\b[a-z]+[a-z0-9]*[a-z]*')

size_table = {
    'BYTE': 8,
    'WORD': 16,
    'DWORD': 32,
    'QWORD': 64
}

# 64 bit registers
regs64 = {
    'rax', 'rbx', 'rcx', 'rdx',
    'rsi', 'rdi', 'rbp', 'rsp',
    'r8', 'r9', 'r10', 'r11',
    'r12', 'r13', 'r14', 'r15'
    }

# 32 bit registers
regs32 = {
    'eax', 'ebx', 'ecx', 'edx',
    'esi', 'edi', 'ebp', 'esp',
    'r8d', 'r9d', 'r10d', 'r11d',
    'r12d', 'r13d', 'r14d', 'r15d'
    }

# 16 bit registers
regs16 = {
    'ax', 'bx', 'cx', 'dx',
    'si', 'di', 'bp', 'sp',
    'r8w', 'r9w', 'r10w', 'r11w',
    'r12w', 'r13w', 'r14w', 'r15w'
    }

# 8 bit registers
regs8 = {
    'al', 'bl', 'cl', 'dl',
    'sil', 'dil', 'bpl', 'spl',
    'r8b', 'r9b', 'r10b', 'r11b',
    'r12b', 'r13b', 'r14b', 'r15b'
    }

def get_access_size(inst):
    size_directive = inst.split()[1]
    # return -1 if it somehow doesn't work
    size = size_table.get(size_directive, None)
    if size is not None:
        return size

    # special case
    fields = inst.split()
    opcode = fields[0]
    if opcode in ('push', 'pop'):
        reg = fields[1]
        if reg in regs64:
            return 64
        if reg in regs32:
            return 32
        if reg in regs16:
            return 16
        if reg in regs8:
            return 8

    return -1

def evaluate_addr(addr):
    registers = reg_re.findall(addr)
    reg_vals = {
        reg : get_register_value(reg)
        for reg in registers }
    # thank god for Intel syntax
    return eval(addr, reg_vals)

def try_extract_mem_access(inst):
  '''
  return the memory access being
  attempted by `inst' (if any)
  '''
  if inst.startswith('lea'):
      return None

  mem = mem_re.match(inst)

  opcode = inst.split()[0]
  if opcode in ('push', 'pop'):
      loc = get_register_value('rsp')
  elif mem is not None:
      addr = mem.group(1)
      loc = evaluate_addr(addr)
  else:
      return None

  size = get_access_size(inst)
  return MemAccess(loc, size)

def get_register_value(reg):
    return int(gdb.parse_and_eval('$' + reg))

def get_cur_inst():
    out = gdb.execute('x/i $pc', to_string=True)
    # `out' will look something like this
    # => 0x4037 <LABEL+12>: sub DWORD PTR [r13+0x0],eax
    return out.split(':')[1].strip()

def dump_accesses(out, user_data_addr, accesses):
    for access in accesses:
        loc, size = access
        # we dump offset relative to `UserData'
        offset = loc - user_data_addr
        out.write('%d,%d\n' % (offset, size))

def run_and_dump_accesses():
    gdb.execute('break '+BEGIN_LABEL)
    gdb.execute('break '+END_LABEL)

    user_data_addr = int(gdb.parse_and_eval('&UserData'))

    # execute until BEGIN_LABEL
    gdb.execute('run')

    accesses = []
    while True:
        inst = get_cur_inst()
        access = try_extract_mem_access(inst)
        if access is not None:
            accesses.append(access)

        # stop when we reach the breakpoint at END_LABEL
        out = gdb.execute('stepi', to_string=True)
        if out.strip().startswith('Breakpoint'):
            break

    with open(OUT_FILE, 'w') as out:
        dump_accesses(out, user_data_addr, accesses)
    exit()

gdb.execute('set pagination no')
gdb.execute('set python print-stack full')
gdb.execute('set disassembly-flavor intel')
run_and_dump_accesses()
