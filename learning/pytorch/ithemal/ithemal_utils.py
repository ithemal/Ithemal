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
    TRANSITIVE_REDUCTION = 1
    TRANSITIVE_CLOSURE = 2
    ADD_LINEAR_EDGES = 3
    ONLY_LINEAR_EDGES = 4
    NO_EDGES = 5

BaseParameters = NamedTuple('BaseParameters', [
    ('data', str),
    ('embed_mode', str),
    ('embed_file', str),
    ('random_edge_freq', float),
    ('predict_log', bool),
    ('no_residual', bool),
    ('edge_ablation_type', Optional[EdgeAblationType]),
    ('embed_size', int),
    ('hidden_size', int),
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
])

BenchmarkParameters = NamedTuple('BenchmarkParameters', [
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('examples', int),
])

def ablate_data(data, edge_ablation_type, random_edge_freq):
    # type: (dt.DataCost, Optional[EdgeAblationType], float) -> None

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
    ablate_data(data, params.edge_ablation_type, params.random_edge_freq)
    return data

def load_model(params, data):
    # type: (BaseParameters, dt.DataCost) -> md.GraphNN

    model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1, use_residual=not params.no_residual)
    model.set_learnable_embedding(mode=params.embed_mode, dictsize=max(data.word2id) + 1, seed=data.final_embeddings)

    return model
