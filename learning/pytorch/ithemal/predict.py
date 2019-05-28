#!/usr/bin/env python

from __future__ import print_function

import argparse
import binascii
import common_libs.utilities as ut
import copy
import data.data_cost as dt
import itertools
import ithemal_utils
import multiprocessing
import os
import subprocess
import sys
import threading
import torch
import warnings

START_MARKER = 'bb6f000000646790'.decode('hex')
END_MARKER = 'bbde000000646790'.decode('hex')

_TOKENIZER = os.path.join(os.environ['ITHEMAL_HOME'], 'data_collection', 'build', 'bin', 'tokenizer')

def load_model_and_data(model_file, model_data_file):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)
        (model, data) = ithemal_utils.load_model_and_data(model_file)

    state_dict = torch.load(model_data_file)
    model_dict = model.state_dict()
    new_model_dict = {k: v for (k, v) in state_dict['model'].items() if k in model_dict}
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    return (model, data)

_fake_intel = '\n'*500

def datum_of_code(data, block_hex, verbose):
    xml = subprocess.check_output([_TOKENIZER, block_hex, '--token'])
    if verbose:
        intel = subprocess.check_output([_TOKENIZER, block_hex, '--intel'])
    else:
        intel = _fake_intel

    data.raw_data = [(-1, -1, intel, xml)]
    data.data = []
    data.prepare_data(fixed=True, progress=False)
    return data.data[-1]

def read_basic_block(fname, data, verbose):
    with open(fname, 'rb') as f:
        code = f.read(-1)
    start_pos = code.index(START_MARKER)
    if start_pos == -1:
        raise ValueError('START MARKER NOT FOUND')

    end_pos = code.index(END_MARKER)
    if end_pos == -1:
        raise ValueError('END MARKER NOT FOUND')

    block_binary = code[start_pos+len(START_MARKER):end_pos]
    return datum_of_code(data, binascii.b2a_hex(block_binary), verbose)

def predict(model, data, fname, verbose):
    datum = read_basic_block(fname, data, verbose)
    if verbose:
        print('='*40)
        print('\n'.join(i.intel for i in datum.block.instrs))
        print('='*40)
    print(model(datum).item())
    model.remove_refs(datum)

def predict_raw(model_arg, data_arg, verbose, parallel):
    input_q = multiprocessing.Queue(1024)
    output_q = multiprocessing.Queue(1024)

    def queue_worker():
        (model, data) = load_model_and_data(model_arg, data_arg)
        while True:
            line = input_q.get()
            if line is None:
                return
            if not line:
                continue
            try:
                datum = datum_of_code(data, line, verbose)
            except:
                output_q.put('{},fail'.format(line))
                continue

            if verbose:
                output_q.put('='*40)
                output_q.put(line)
                output_q.put('\n'.join(i.intel for i in datum.block.instrs))
                output_q.put('='*40)
            output_q.put('{},{}'.format(line, model(datum).item()))
            model.remove_refs(datum)

    def output_writer():
        while True:
            line = output_q.get()
            if line is None:
                return
            print(line)

    workers = [multiprocessing.Process(target=queue_worker) for _ in range(parallel)]
    for w in workers:
        w.start()
    output_worker = multiprocessing.Process(target=output_writer)
    output_worker.start()
    try:
        while True:
            line = raw_input().strip()
            input_q.put(line)
    except EOFError:
        for _ in range(parallel):
            input_q.put(None)
    for w in workers:
        w.join()
    output_q.put(None)
    output_worker.join()

def main():
    parser = argparse.ArgumentParser(description='Analyze a basic block')
    parser.add_argument('--model', help='Model architecture to use', required=True)
    parser.add_argument('--model-data', help='Model data to use', required=True)
    parser.add_argument('--verbose', help='Whether to be verbose', action='store_true', default=False)
    parser.add_argument('--parallel', help='How many parallel threads to run', type=int, default=1)

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--raw-stdin', help='Whether to read newline-separated raw hex from stdin', action='store_true', default=False)
    input_group.add_argument(
        '--files',
        help='Binary files to analyze. Relevant basic block must be started by 0x{} and terminated by 0x{}'.format(
            binascii.b2a_hex(START_MARKER),
            binascii.b2a_hex(END_MARKER)
        ),
        nargs='+',
    )
    args = parser.parse_args()

    if args.raw_stdin:
        predict_raw(args.model, args.model_data, args.verbose, args.parallel)
    else:
        (model, data) = load_model_and_data(args.model, args.model_data)
        for fname in args.files:
            predict(model, data, fname, args.verbose)

if __name__ == '__main__':
    main()
