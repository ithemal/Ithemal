from typing import NamedTuple

LossReportMessage = NamedTuple('LossReportMessage', [
    ('rank', int),
    ('loss', float),
    ('n_items', int),
])

EpochAdvanceMessage = NamedTuple('EpochAdvanceMessage', [
    ('epoch', int),
])
