#!/usr/bin/env python

from __future__ import print_function

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
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence

try:
    import cPickle as pickle
except ImportError:
    import pickle # type: ignore

_DIRNAME = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_DIRNAME, os.pardir, 'inputs', 'augmentations')
_DEFAULT_DUP_TEMPLATE = os.path.join(_DIRNAME, 'eu_dup_template.json')

_time_str = None # type: Optional[str]
def time_str(): # type: () -> str
    global _time_str
    if _time_str is None:
        _time_str = time.strftime('%Y-%m-%d.%H-%M-%S')
    return _time_str

def save_object(obj, name): # type: (Any, str) -> None
    with open(os.path.join(_DATA_DIR, '{}_{}.pkl'.format(name, time_str())), 'wb') as f:
        pickle.dump(obj, f)

def execute_sql(commands): # type: (List[str]) -> None
    cnx = ut.create_connection()
    for com in commands:
        ut.execute_query(cnx, com, False)
    cnx.commit()


def read_dataset(data_file, embedding_file): # type: (str, str) -> dt.DataInstructionEmbedding
    data = dt.DataInstructionEmbedding()

    data.raw_data = torch.load(data_file)
    data.set_embedding(embedding_file)
    data.read_meta_data()
    data.prepare_data()
    data.generate_datasets()
    return data


AugmentationMap = Dict[dt.DataItem, Iterable[Sequence[ut.Instruction]]]

def gen_permutations(
        full_data,
        desired_n_perms=None,
        max_block_size=None,
        min_perms_per_block=None,
        max_perms_per_block=None
):
    # type: (dt.DataInstructionEmbedding, Optional[int], Optional[int], Optional[int], Optional[int]) -> AugmentationMap
    data = set(full_data.data)
    perms = {} # type: AugmentationMap

    n_perms_gen = 0

    pbar = tqdm(total=(desired_n_perms or len(data)))
    while data and (desired_n_perms is None or n_perms_gen < desired_n_perms):
        datum = data.pop()
        block = datum.block
        if max_block_size and len(block.instrs) > max_block_size:
            continue
        if max_perms_per_block:
            reorderings = set() # type: Set[Sequence[ut.Instruction]]
            n_tries = 0
            while len(reorderings) < max_perms_per_block and n_tries < max_perms_per_block * 2:
                m_reorderings = block.gen_reorderings(single_perm=True)
                assert len(m_reorderings) == 1
                reorderings.add(tuple(m_reorderings[0]))
                n_tries += 1
        else:
            reorderings = set(map(tuple, block.gen_reorderings()))
        if min_perms_per_block and len(reorderings) < min_perms_per_block:
            continue
        perms[datum] = reorderings
        n_perms_gen += len(reorderings)
        pbar.update(len(reorderings) if desired_n_perms else 1)
    pbar.close()

    return perms

def gen_duplicated_instructions(full_data, max_dups):
    # type: (dt.DataInstructionEmbedding, int) -> AugmentationMap

    data = set(full_data.data)
    perms = {} # type: AugmentationMap

    pbar = tqdm(total=len(data))
    while data:
        datum = data.pop()
        block = datum.block
        reorderings = ut.generate_duplicates(block.instrs, max_dups)
        if reorderings:
            perms[datum] = reorderings
        pbar.update(1)
    pbar.close()

    return perms

def gen_sql_commands_of_augs(augs, table_name): # type: (AugmentationMap, str) -> List[str]
    sql_commands = []
    sql_commands.append('''CREATE TABLE {} (
        aug_id int(32) NOT NULL AUTO_INCREMENT,
        code_id int(32) NOT NULL,
        code_intel TEXT NOT NULL,
        code_token TEXT NOT NULL,
        PRIMARY KEY (aug_id),
        CONSTRAINT {}_idfk_1 FOREIGN KEY (code_id) REFERENCES code(code_id)
    );'''.format(table_name, table_name))

    def format_insert_command(values): # List[str] -> str
        return 'INSERT INTO {} (code_id, code_intel, code_token) VALUES ({});'.format(
            table_name,
            ','.join(values),
        )

    for dataitem in tqdm(augs):
        for aug in augs[dataitem]:
            tokens = []
            for i in aug:
                tokens.append(i.opcode)
                tokens.append(-1)
                tokens.extend(i.srcs)
                tokens.append(-1)
                tokens.extend(i.dsts)
                tokens.append(-1)

            values = [
                str(dataitem.code_id),
                "'{}'".format('\n'.join(i.intel for i in aug)),
                "'{}'".format(','.join(map(str, tokens))),
            ]
            sql_commands.append(format_insert_command(values))

    return sql_commands

def main(): # type: () -> None
    parser = argparse.ArgumentParser(description='Supplement dataset')
    parser.add_argument('--data', type=str, required=True, help='Block data file to use (e.g. inputs/data/time_skylake.data')
    parser.add_argument('--embedding', type=str, required=True, help='Token embedding file to use (e.g. inputs/embeddings/code_delim.emb)')
    parser.add_argument('--table-name', type=str, required=True, help='Table to write augmentations to (will be freshly created)')

    parser.add_argument('--execute-sql', action='store_true', default=False)
    parser.add_argument('--store-sql', action='store_true', default=False)
    parser.add_argument('--optimize-sql', action='store_true', default=False)

    subparsers = parser.add_subparsers(dest='command')

    perms_parser = subparsers.add_parser('permutations')
    perms_parser.add_argument('--desired-n-perms', default='all')
    perms_parser.add_argument('--max-block-size', type=int, default=None, help='Maximum block size to attempt to generate permutations for. Default none')
    perms_parser.add_argument('--min-perms-per-block', type=int, default=None, help='Minimum number of permutations to include when generating permutations (otherwise throw out block)')
    perms_parser.add_argument('--max-perms-per-block', type=int, default=None, help='Maximum numnber of permutations to include when generating permuations.')

    ports_parser = subparsers.add_parser('ports')
    ports_parser.add_argument('--dup-template', type=str, default=_DEFAULT_DUP_TEMPLATE)
    ports_parser.add_argument('--max-dups', type=int, default=10, help='Max number of times to duplicate a given instruction')

    args = parser.parse_args()

    data = read_dataset(args.data, args.embedding)

    if args.command == 'permutations':
        if args.desired_n_perms == 'all':
            desired_n_perms = None
        elif args.desired_n_perms == 'equal':
            desired_n_perms = len(data.data)
        else:
            desired_n_perms = int(args.desired_n_perms)

        augs = gen_permutations(
            data,
            desired_n_perms=desired_n_perms,
            max_block_size=args.max_block_size,
            min_perms_per_block=args.min_perms_per_block,
            max_perms_per_block=args.max_perms_per_block,
        )
    else:
        augs = gen_duplicated_instructions(data, args.max_dups)

    sql_commands = gen_sql_commands_of_augs(augs, args.table_name)

    if args.optimize_sql:
        sql_commands.insert(0, 'SET autocommit=0;')
        sql_commands.insert(1, 'SET unique_checks=0;')
        sql_commands.insert(2, 'SET foreign_key_checks=0;')
        sql_commands.append('COMMIT;')
        sql_commands.append('SET unique_checks=1;')
        sql_commands.append('SET foreign_key_checks=1;')
        sql_commands.append('SET autocommit=1;')

    if args.store_sql:
        with open(os.path.join(_DATA_DIR, 'table_{}.sql'.format(time_str())), 'w') as f:
            print('\n'.join(sql_commands), file=f)

    if args.execute_sql:
        execute_sql(sql_commands)

if __name__ == '__main__':
    main()
