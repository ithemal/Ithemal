import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

from enum import Enum
import torch
from typing import Any, Callable, List, Optional, Iterator, Tuple, NamedTuple, Union

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
    ('dag_reduction', md.ReductionType),
    ('edge_ablation_types', List[EdgeAblationType]),
    ('embed_size', int),
    ('hidden_size', int),
    ('linear_embeddings', bool),
    ('use_rnn', bool),
    ('rnn_type', md.RnnType),
    ('rnn_hierarchy_type', md.RnnHierarchyType),
    ('rnn_connect_tokens', bool),
    ('rnn_skip_connections', bool),
    ('rnn_learn_init', bool),
    ('no_mem', bool),
    ('linear_dependencies', bool),
    ('flat_dependencies', bool),
    ('dag_nonlinearity', md.NonlinearityType),
    ('dag_nonlinearity_width', int),
    ('dag_nonlinear_before_max', bool),
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
    ('lr_decay_rate', float),
])

BenchmarkParameters = NamedTuple('BenchmarkParameters', [
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('examples', int),
])

PredictorDump = NamedTuple('PredictorDump', [
    ('model', md.AbstractGraphModule),
    ('dataset_params', Any),
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
    data = dt.load_dataset(params.data)

    def filter_data(filt):
        # type: (Callable[[dt.DataItem], bool]) -> None
        data.data = [d for d in data.data if filt(d)]
        data.train = [d for d in data.train if filt(d)]
        data.test = [d for d in data.test if filt(d)]

    if params.no_mem:
        filter_data(lambda d: not d.block.has_mem())

    ablate_data(data, params.edge_ablation_types, params.random_edge_freq)

    if params.linear_dependencies:
        filter_data(lambda d: d.block.has_linear_dependencies())

    if params.flat_dependencies:
        filter_data(lambda d: d.block.has_no_dependencies())

    return data

def load_model(params, data):
    # type: (BaseParameters, dt.DataCost) -> md.AbstractGraphModule

    if params.use_rnn:
        rnn_params = md.RnnParameters(
            embedding_size=params.embed_size,
            hidden_size=params.hidden_size,
            num_classes=1,
            connect_tokens=params.rnn_connect_tokens,
            skip_connections=params.rnn_skip_connections,
            hierarchy_type=params.rnn_hierarchy_type,
            rnn_type=params.rnn_type,
            learn_init=params.rnn_learn_init,
        )
        model = md.RNN(rnn_params)
    else:
        model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1,
                           use_residual=not params.no_residual, linear_embed=params.linear_embeddings,
                           use_dag_rnn=not params.no_dag_rnn, reduction=params.dag_reduction,
                           nonlinear_type=params.dag_nonlinearity, nonlinear_width=params.dag_nonlinearity_width,
                           nonlinear_before_max=params.dag_nonlinear_before_max,
        )

    model.set_learnable_embedding(mode=params.embed_mode, dictsize=628 or max(data.hot_idx_to_token) + 1)

    return model

def dump_model_and_data(model, data, fname):
    # type: (md.AbstractGraphMode, dt.DataCost, str) -> None
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass
    torch.save(PredictorDump(
        model=model,
        dataset_params=data.dump_dataset_params(),
    ), fname)

def load_model_and_data(fname):
    # type: (str) -> (md.AbstractGraphMode, dt.DataCost)
    dump = torch.load(fname)
    data = dt.DataInstructionEmbedding()
    data.read_meta_data()
    data.load_dataset_params(dump.dataset_params)
    return (dump.model, data)
