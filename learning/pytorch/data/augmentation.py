#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

# this script doesn't need matplotlib; this line fixes interactive use of this script.
import matplotlib
matplotlib.use('Agg')

import argparse
import common_libs.utilities as ut
import data_cost as dt
import os
import time
import torch
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle

def read_dataset(data_file, embedding_file):
    data = dt.DataInstructionEmbedding()

    data.raw_data = torch.load(data_file)
    data.set_embedding(embedding_file)
    data.read_meta_data()
    data.prepare_data()
    data.generate_datasets()
    return data

memory_trace = lambda x: [i for i in x if i.has_mem()]

_DIRNAME = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_DIRNAME, os.pardir, 'data', 'augmentations')

def _curr_time_str():
    return time.strftime('%Y-%m-%d.%H-%M-%S')

def gen_permutations(full_data):
    data = set(full_data.data)
    n_perms_gen = 0
    perms = {}
    desired_n_perms = len(full_data.data)
    block_size_cutoff = 10

    with tqdm(total=desired_n_perms) as pbar:
        while n_perms_gen < desired_n_perms:
            datum = data.pop()
            block = datum.block
            if len(block.instrs) > block_size_cutoff:
                continue
            reorderings = set(map(tuple, block.gen_reorderings()))
            perms[datum] = reorderings
            n_perms_gen += len(reorderings)
            pbar.update(len(reorderings))

    return perms

def write_perms():
    try:
        os.makedirs(_DATA_DIR)
    except OSError:
        pass

    with open(os.path.join(_DATA_DIR, 'permutations_{}'.format(_curr_time_str())), 'wb') as f:
        pickle.dump(perms, f)

def read_perms(perms_file):
    with open(perms_file, 'rb') as f:
        return pickle.load(f)

def save_perms_to_table(perms, table_name):
    cnx = ut.create_connection()
    sql = '''CREATE TABLE {} (
    perm_id int(32) NOT NULL AUTO_INCREMENT,
    code_id int(32) NOT NULL,
    code_intel TEXT NOT NULL,
    code_token TEXT NOT NULL,
    PRIMARY KEY (perm_id),
    CONSTRAINT {}_idfk_1 FOREIGN KEY (code_id) REFERENCES code(code_id)
    );'''.format(table_name, table_name)
    ut.execute_query(cnx, sql, False)

    values = []
    for dataitem in tqdm(perms):
        for perm in perms[dataitem]:
            tokens = []
            for i in perm:
                tokens.append(i.opcode)
                tokens.append(-1)
                tokens.extend(i.srcs)
                tokens.append(-1)
                tokens.extend(i.dsts)
                tokens.append(-1),
            values_str = '({}, {}, {})'.format(
                dataitem.code_id,
                "'{}'".format(','.join(map(str, tokens))),
                "'{}'".format('\n'.join(i.intel for i in perm)),
            )
            values.append(values_str)
            sql = 'INSERT INTO {} (code_id, code_intel, code_token) VALUES {}'.format(
                table_name,
                ','.join(values),
            )
            ut.execute_query(cnx, sql, False)
    cnx.commit()


def main():
    parser = argparse.ArgumentParser(description='Supplement dataset')
    parser.add_argument('--data', type=str, required=True, help='Block data file to use (e.g. inputs/data/time_skylake.data')
    parser.add_argument('--embedding', type=str, required=True, help='Token embedding file to use (e.g. inputs/embeddings/code_delim.emb)')
    parser.add_argument('--table-name', type=str, required=True, help='Table to write permutations to (will be freshly created)')
    parser.add_argument('--save-perms', action='store_true', default=False)
    args = parser.parse_args()

    data = read_dataset(args.data, args.embedding)
    perms = gen_permutations(data)
    if args.save_perms:
        save_perms(perms)
    save_perms_to_table(perms, args.table_name)


if __name__ == '__main__':
    main()
