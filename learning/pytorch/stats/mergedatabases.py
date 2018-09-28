from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

import common_libs.utilities as ut
import models.graph_models as md
import models.train as tr
import data.data_cost as dt
import word2vec.word2vec as w2v

from tqdm import tqdm
import subprocess
import os
import re
import time
import argparse
import statistics
import pickle
import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


if __name__ == '__main__':

    main_cnx = ut.create_connection('costmodel')

    cnx = ut.create_connection('costmodel_skylake')
    sql = 'select code_id, arch, kind, time, count from times where arch=2'
    rows = ut.execute_query(cnx, sql, True)

    for row in tqdm(rows):
        
        sql = 'INSERT INTO times (code_id, arch, kind, time) VALUES ('
        sql += str(row[0]) + ','
        sql += str(row[1]) + ','
        sql += '\'' + str(row[2]) + '\','
        sql += str(row[3]) + ')'

        ut.execute_query(main_cnx, sql, False)
        main_cnx.commit()

    cnx.close()
    main_cnx.close()

