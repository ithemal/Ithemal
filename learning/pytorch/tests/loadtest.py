import numpy as np
import common_libs.utilities as ut
import torch
import mysql.connector
import argparse
import matplotlib

matplotlib.use('Agg')

import sys
import common.common_libs.utilities as ut
import numpy as np
import word2vec.word2vec as w2v


if __name__ == "__main__":

    #commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',required=True)
    args = parser.parse_args(sys.argv[1:])

    with open(args.embed_file,'r') as f:
        (final_embeddings,_,_) = torch.load(f)

    #create database connection
    cnx = ut.create_connection('costmodel')

    embedder = w2v.Word2Vec(num_steps = 10000)

    raw_data = ut.get_data(cnx,args.format,[])
    token_data = list()
    for row in raw_data:
        token_data.extend(row[0])
    print len(token_data)

    offsets_filename = '/data/scratch/charithm/projects/cmodel/database/offsets.txt'
    sym_dict, mem_start = ut.get_sym_dict(offsets_filename)
    offsets = ut.read_offsets(offsets_filename)

    data = embedder.generate_dataset(token_data,sym_dict,mem_start)

    embedder.print_associated_words(final_embeddings,200,sym_dict,mem_start)
    embedder.plot_with_labels(final_embeddings,200,sym_dict,mem_start)





