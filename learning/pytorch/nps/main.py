#!/usr/bin/env python
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import matplotlib as mpl
mpl.use('Agg')

import common_libs.utilities as ut
import data.data_cost as dt
import neural_processor_simulator

def main():
    root = os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch')
    data = dt.load_dataset(
        os.path.join(root, 'inputs', 'embeddings', 'code_delim.emb'),
        os.path.join(root, 'saved', 'time_skylake_1217.data')
    )
    model = neural_processor_simulator.NeuralProcessorSimulator()
    trainer = neural_processor_simulator.Trainer(model, data.train, 'nps')

if __name__ == '__main__':
    main()
