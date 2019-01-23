import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

from enum import Enum
import torch
from typing import Callable, List, Optional, Iterator, Tuple, NamedTuple, Union

import data.data_cost as dt
import models.graph_models as md
import models.train as tr

class EdgeAblationType(Enum):
    TRANSITIVE_REDUCTION = 'transitive-reduction'
    TRANSITIVE_CLOSURE = 'transitive-closure'
    ADD_LINEAR_EDGES = 'add-linear-edges'
    ONLY_LINEAR_EDGES = 'only-linear-edges'
    NO_EDGES = 'no-edges'

BaseParameters = NamedTuple('BaseParameters', [
    ('data', str),
    ('embed_mode', str),
    ('embed_file', str),
    ('random_edge_freq', float),
    ('predict_log', bool),
    ('no_residual', bool),
    ('no_dag_rnn', bool),
    ('edge_ablation_types', List[EdgeAblationType]),
    ('embed_size', int),
    ('hidden_size', int),
    ('linear_embeddings', bool),
    ('use_rnn', bool),
    ('rnn_hierarchical', bool),
    ('rnn_connect_tokens', bool),
    ('rnn_dense', bool),
])

TrainParameters = NamedTuple('TrainParameters', [
    ('experiment_name', str),
    ('experiment_time', str),
    ('load_file', Optional[str]),
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('decay_trainers', bool),
    ('weight_decay', float),
    ('initial_lr', float),
    ('decay_lr', bool),
    ('epochs', int),
    ('split', Union[int, List[float]]),
    ('optimizer', tr.OptimizerType),
    ('momentum', float),
    ('nesterov', bool),
    ('weird_lr', bool),
])

BenchmarkParameters = NamedTuple('BenchmarkParameters', [
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('examples', int),
])

def ablate_data(data, edge_ablation_types, random_edge_freq):
    # type: (dt.DataCost, List[EdgeAblationType], float) -> None

    for edge_ablation_type in edge_ablation_types:
        if edge_ablation_type == EdgeAblationType.TRANSITIVE_REDUCTION:
            for data_item in data.data:
                data_item.block.transitive_reduction()
        elif edge_ablation_type == EdgeAblationType.TRANSITIVE_CLOSURE:
            for data_item in data.data:
                data_item.block.transitive_closure()
        elif edge_ablation_type == EdgeAblationType.ADD_LINEAR_EDGES:
            for data_item in data.data:
                data_item.block.linearize_edges()
        elif edge_ablation_type == EdgeAblationType.ONLY_LINEAR_EDGES:
            for data_item in data.data:
                data_item.block.remove_edges()
                data_item.block.linearize_edges()
        elif edge_ablation_type == EdgeAblationType.NO_EDGES:
            for data_item in data.data:
                data_item.block.remove_edges()

    if random_edge_freq > 0:
        for data_item in data.data:
            data_item.block.random_forward_edges(random_edge_freq / len(data_item.block.instrs))

def load_data(params):
    # type: (BaseParameters) -> dt.DataCost
    data = dt.load_dataset(params.embed_file, data_savefile=params.data)
    ablate_data(data, params.edge_ablation_types, params.random_edge_freq)
    return data

def load_model(params, data):
    # type: (BaseParameters, dt.DataCost) -> md.AbstractGraphModule

    if params.use_rnn:
        model = md.RNNs(
            embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1,
            use_hierarchical=params.rnn_hierarchical, connect_tokens=params.rnn_connect_tokens, dense_hierarchical=params.rnn_dense,
        )
    else:
        model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1,
                           use_residual=not params.no_residual, linear_embed=params.linear_embeddings,
                           use_dag_rnn=not params.no_dag_rnn,
        )

    model.set_learnable_embedding(mode=params.embed_mode, dictsize=max(data.word2id) + 1, seed=data.final_embeddings)

    return model
