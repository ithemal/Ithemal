'''ZMQ, Pickle, and NamedTuples are together broken enough that you
can't have the __main__ module declare a NamedTuple which is sent through ZMQ.

To remedy that, we declare all NamedTuples here

'''

from ithemal_utils import BaseParameters, TrainParameters
from typing import Any, Dict, List, Iterator, Tuple, Type, Union, NamedTuple

TrainerInitializeReq = NamedTuple('TrainerInitializeReq', [
    ('rank', int),
])
TrainerInitializeResp = NamedTuple('TrainerInitializeResp', [
    ('base_params', BaseParameters),
    ('train_params', TrainParameters),
])

TrainerDataReq = NamedTuple('TrainerDataReq', [
    ('rank', int),
])
TrainerDataResp = NamedTuple('TrainerDataResp', [
    ('model_tensor_params', Any),
    ('trainer_tensor_params', Any),
])

# ------------------------------

TrainerStepReq = NamedTuple('TrainerStepReq', [
    ('rank', int),
])
WaitResp = NamedTuple('WaitResp', [])
KillResp = NamedTuple('KillResp', [])
SetLrResp = NamedTuple('SetLrResp', [
    ('new_lr', float),
])
ShuffleDataResp = NamedTuple('ShuffleDataResp', [
    ('random_state', object),
])
RunTrainerResp = NamedTuple('RunTrainerResp', [
    ('partition', Tuple[int, int]),
])

# ------------------------------

TrainerLossReq = NamedTuple('TrainerLossReq', [
    ('rank', int),
    ('loss', float),
    ('n_items', int),
])
TrainerLossResp = NamedTuple('TrainerLossResp', [])

# ------------------------------

TrainerDeathReq = NamedTuple('TrainerDeathReq', [
    ('rank', int),
    ('partition_remainder', Tuple[int, int]),
])
TrainerDeathResp = NamedTuple('TrainerDeathResp', [])
