#this creates datafiles which are needed for consistency. Anything that can be calculated is left for each individual data processor

import sys
sys.path.append('..')

import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
matplotlib.use('Agg')

import common_libs.utilities as ut
import graphs as gr
import numpy as np
import torch

import data.data as dt_abs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)

    parser.add_argument('--database',action='store',default='text',type=str)
    parser.add_argument('--user',action='store',type=str)
    parser.add_argument('--password',action='store',type=str)
    parser.add_argument('--port',action='store',type=int)

    args = parser.parse_args(sys.argv[1:])

    cnx = ut.create_connection(database=args.database, user=args.user, password=args.password, port=args.port)

    raw_data = ut.get_data(cnx, args.format, [])

    torch.save(raw_data,'saved/ins.data')

    data = dt_abs.Data()
    data.read_meta_data()

    data.generate_costdict(20)
    torch.save(data.costs, '../saved/cost20.data')

    data.generate_costdict(50)
    torch.save(data.costs, '../saved/cost50.data')

    data.generate_costdict(100)
    torch.save(data.costs, '../saved/cost100.data')

