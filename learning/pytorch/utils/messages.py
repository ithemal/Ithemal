from typing import NamedTuple, Tuple, Union

LossReportMessage = NamedTuple('LossReportMessage', [
    ('rank', int),
    ('loss', float),
    ('n_items', int),
])

EpochAdvanceMessage = NamedTuple('EpochAdvanceMessage', [
    ('epoch', int),
    ('n_trainers', int),
])

TrainerDeathMessage = NamedTuple('TrainerDeathMessage', [
    ('remaining_partition', Tuple[int, int]),
])

Message = Union[LossReportMessage, EpochAdvanceMessage, TrainerDeathMessage]
