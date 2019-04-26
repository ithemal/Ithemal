#!/usr/bin/env python

from gevent import monkey; monkey.patch_all()

import argparse
import functools
import subprocess
import gevent
import os
import tempfile

_IACA_HEADER = "7f454c4602010100000000000000000001003e000100000000000000000000000000000000000000100100000000000000000000400000000000400004000100bb6f000000646790"
_IACA_TAIL = "bbde000000646790"

_LLVM_BODY = '''        .text
        .att_syntax
        .globl          main
main:
        # LLVM-MCA-BEGIN test
{}
        # LLVM-MCA-END test
    '''

_IACA = os.path.join(os.environ['ITHEMAL_HOME'], 'timing_tools', 'iaca-bin')
_LLVM = os.path.join(os.environ['ITHEMAL_HOME'], 'timing_tools', 'llvm-build', 'bin', 'llvm-mca')
_DISASSMBLER = os.path.join(os.environ['ITHEMAL_HOME'], 'data_collection', 'disassembler', 'build', 'disassemble')


def time_llvm(arch, code):
    with tempfile.NamedTemporaryFile() as f:
        disassembler = subprocess.Popen([_DISASSMBLER, '-att'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        (output, _) = disassembler.communicate(code)
        f.write(_LLVM_BODY.format(output))
        f.flush()
        output = subprocess.check_output([_LLVM, '-march=x86', '-mcpu={}'.format(arch), f.name])
        total_cycles_line = output.split('\n')[5]
        cycles = total_cycles_line.split()[2]
        return float(cycles)

def time_iaca(arch, code):
    with tempfile.NamedTemporaryFile() as f:
        f.write('{}{}{}'.format(_IACA_HEADER, code, _IACA_TAIL).decode('hex'))
        f.flush()
        output = subprocess.check_output([_IACA, '-arch', arch, '-reduceout', f.name])
        txput_line = output.split('\n')[3]
        txput = txput_line.split()[2]
        return float(txput) * 100

def time_code_ids(code_ids, timer):
    print('START counting {}'.format(len(code_ids)))
    # get code
    mysql = subprocess.Popen(['mysql', '-N'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (out, _) = mysql.communicate('SELECT code_raw FROM code WHERE code_id IN ({});\n'.format(','.join(map(str, code_ids))))
    jobs = {code_id: gevent.spawn(timer, code_raw) for (code_id, code_raw) in zip(code_ids, out.split('\n'))}
    gevent.joinall(jobs.values(), timeout=240)
    print('END counting {}'.format(len(code_ids)))
    return {code_id: jobs[code_id].value for code_id in jobs}

iaca_kind = (2, time_iaca, {'haswell': 'HSW', 'broadwell': 'BDW', 'skylake': 'SKL'})
llvm_kind = (3, time_llvm, {'haswell': 'haswell', 'broadwell': 'broadwell', 'skylake': 'skylake', 'nehalem': 'nehalem'})

_kind_map = {
    'iaca': iaca_kind,
    'llvm': llvm_kind,
}

_arch_map = {
    'haswell': 1,
    'skylake': 2,
    'broadwell': 3,
    'nehalem': 4
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', type=str)
    parser.add_argument('kind')
    parser.add_argument('code_id', type=int, nargs='+')
    args = parser.parse_args()

    (kind_id, timer_func, arch_dict) = _kind_map[args.kind]
    arch_id = _arch_map[args.arch]
    timer = functools.partial(timer_func, arch_dict[args.arch])

    times = time_code_ids(args.code_id, timer)

    mysql = subprocess.Popen(['mysql'], stdin=subprocess.PIPE)
    values = ','.join(map(str, ((code_id, arch_id, kind_id, speed) for (code_id, speed) in times.items() if speed is not None)))
    mysql.communicate('INSERT INTO time (code_id, arch_id, kind_id, cycle_count) VALUES {};\n'.format(values))


if __name__ == '__main__':
    main()
