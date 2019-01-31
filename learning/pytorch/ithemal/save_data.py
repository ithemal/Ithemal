import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import argparse
import common_libs.utilities as ut
import data.data_cost as dt
import torch
from typing import Optional

def save_data(savefile, arch, format, database=None, config=None):
    # type: (str, int, str, Optional[str], Optional[str]) -> None

    if config is None:
        cnx = ut.create_connection(database=database)
    else:
        cnx = ut.create_connection_from_config(database=database, config_file=config)

    data = dt.DataInstructionEmbedding()
    data.extract_data(cnx, format, ['code_id','code_intel'])
    data.get_timing_data(cnx, arch)

    torch.save(data.raw_data, savefile)

def main():
    # type: () -> None
    parser = argparse.ArgumentParser('Save data from SQL to disk')
    parser.add_argument('dest', help='Location to save the data to')
    parser.add_argument('--format', default='text', help='Format to save data in')
    parser.add_argument('--arch', type=int, help='Architecture of data to pull', required=True)
    parser.add_argument('--database', help='Database to pull from (if not default)')
    parser.add_argument('--config', help='Database configuration to use (if not deafult)')

    args = parser.parse_args()

    save_data(args.dest, args.arch, args.format, database=args.database, config=args.config)

if __name__ == '__main__':
    main()
