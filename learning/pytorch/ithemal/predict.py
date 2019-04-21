#!/usr/bin/env python

import argparse
import binascii
import common_libs.utilities as ut
import data.data_cost as dt
import os
import subprocess
import torch
import ithemal_utils

START_MARKER = b'\xab\xa6\x58'
END_MARKER = b'\x9e\x87\x0f'

_TOKENIZER = os.path.join(os.environ['ITHEMAL_HOME'], 'data_collection', 'build', 'bin', 'tokenizer')

def load_model_and_data(model_file, model_data_file):
    (model, data) = ithemal_utils.load_model_and_data(model_file)

    state_dict = torch.load(model_data_file)
    model_dict = model.state_dict()
    new_model_dict = {k: v for (k, v) in state_dict['model'].items() if k in model_dict}
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    return (model, data)

def read_basic_block(fname, data):
    with open(fname, 'rb') as f:
        code = f.read(-1)
    start_pos = code.index(START_MARKER)
    if start_pos == -1:
        raise ValueError('START MARKER NOT FOUND')

    end_pos = code.index(END_MARKER)
    if end_pos == -1:
        raise ValueError('END MARKER NOT FOUND')

    block_binary = code[start_pos+len(START_MARKER):end_pos]
    xml = subprocess.check_output([_TOKENIZER, binascii.b2a_hex(block_binary), '--token'])
    intel = subprocess.check_output([_TOKENIZER, binascii.b2a_hex(block_binary), '--intel'])

    data.raw_data = [(-1, -1, intel, xml)]
    data.prepare_data(fixed=True, progress=False)
    return data.data[-1]

def predict(model, data, fname, verbose):
    datum = read_basic_block(fname, data)
    if verbose:
        print('='*40)
        print('\n'.join(i.intel for i in datum.block.instrs))
    print(model(datum).item())
    model.remove_refs(datum)

def main():
    parser = argparse.ArgumentParser(description='Analyze a basic block')
    parser.add_argument(
        'files',
        help='Binary files to analyze. Relevant basic block must be started by 0x{} and terminated by 0x{}'.format(
            binascii.b2a_hex(START_MARKER),
            binascii.b2a_hex(END_MARKER)
        ),
        nargs='+',
    )
    parser.add_argument('--model', help='Model architecture to use', required=True)
    parser.add_argument('--model-data', help='Model data to use', required=True)
    parser.add_argument('--verbose', help='Whether to be verbose', action='store_true', default=False)
    args = parser.parse_args()

    (model, data) = load_model_and_data(args.model, args.model_data)
    for fname in args.files:
        predict(model, data, fname, args.verbose)

if __name__ == '__main__':
    main()
