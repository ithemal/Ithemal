from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import common_libs.utilities as ut
import data.data_cost as dt
import functools
from pprint import pprint
import random
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, NamedTuple

SLOT_WIDTH = 32
NUM_SLOTS = 8

SimulatorInput = NamedTuple('SimulatorInput', [
    ('slot_vector', torch.tensor),
    ('instruction_vector', torch.tensor),
])
SimulatorResult = NamedTuple('SimulatorResult', [
    ('wait_time', torch.tensor),
    ('write_head', torch.tensor),
    ('write_state', torch.tensor),
    ('write_time', torch.tensor),
])
ModelRunResult = NamedTuple('ModelRunResult', [
    ('prediction', torch.tensor),
    ('loss', torch.tensor),
    ('slots', List['Slot']),
])

class Slot(object):
    def __init__(self):
        # type: () -> None
        self.state = torch.randn(SLOT_WIDTH, requires_grad=True)
        self.remaining_time = torch.zeros(1, requires_grad=True)

    def mutate(self, new_state, additional_time):
        # type: (torch.tensor, torch.tensor) -> torch.tensor
        self.state = self.state * (self.remaining_time > 0).float() + new_state
        old_time = self.remaining_time
        self.remaining_time = self.remaining_time + additional_time
        return old_time[0]

    def step(self, time):
        # type: (torch.tensor) -> None
        self.remaining_time = torch.clamp(self.remaining_time - time, min=0)

    def read(self):
        # type: () -> torch.tensor
        return torch.cat([self.remaining_time, self.state * (self.remaining_time > 0).float()])

def cat_embedder(emb_dim, max_n_srcs, max_n_dsts):
    # type: (int, int, int) -> Callable[[ut.Instruction], torch.tensor]
    sym_dict, _ = ut.get_sym_dict()
    embedder = torch.nn.Embedding(len(sym_dict), emb_dim)
    clamp = lambda x: x if x < len(sym_dict) else len(sym_dict) - 1

    def get_emb_list(arr, length):
        # type: (List[int], int) -> List[torch.tensor]
        assert len(arr) <= length
        real = [embedder(torch.tensor(clamp(val))) for val in arr]
        zeros = [torch.zeros(emb_dim) for _ in range(length - len(arr))]
        return real + zeros

    def embed(instr):
        # type: (ut.Instruction) -> torch.tensor
        opc = embedder(torch.tensor(instr.opcode))
        srcs = get_emb_list(instr.srcs, max_n_srcs)
        dsts = get_emb_list(instr.dsts, max_n_dsts)
        return torch.cat([opc] + srcs + dsts)

    return embed


class NeuralProcessorSimulator(nn.Module):
    def __init__(self):
        # type: () -> None
        super(NeuralProcessorSimulator, self).__init__()
        self.embedder = cat_embedder(128, 3, 3)
        self.instr_vec_emb = nn.Linear(128*7, 128)
        self.slot_vec_emb = nn.Linear((1+SLOT_WIDTH)*NUM_SLOTS, 128)
        self.wait_time_out = nn.Linear(256, 1)
        self.write_head_out = nn.Linear(256, NUM_SLOTS)
        self.write_state_out = nn.Linear(256, SLOT_WIDTH)
        self.write_time_out = nn.Linear(256, 1)

    def forward(self, instr_vec, slot_vec):
        # type: (torch.tensor, torch.tensor) -> SimulatorResult
        instr_vec = F.relu(self.instr_vec_emb(instr_vec))
        slot_vec = F.relu(self.slot_vec_emb(slot_vec))
        concat = torch.cat([instr_vec, slot_vec])

        wait_time = self.wait_time_out(concat).abs()
        write_head = F.softmax(self.write_head_out(concat), dim=0)
        write_state = self.write_state_out(concat)
        write_time = self.write_time_out(concat).abs()

        return SimulatorResult(
            wait_time=wait_time,
            write_head=write_head,
            write_state=write_state,
            write_time=write_time,
        )

def run_on_data(model, block, actual, debug=False):
    # type: (nn.Module, ut.BasicBlock, float, bool) -> ModelRunResult
    slots = [Slot() for _ in range(NUM_SLOTS)]
    schedule_loss = torch.tensor(0., requires_grad=True)
    wait_time = torch.tensor(0., requires_grad=True)

    for instr in block.instrs:
        slot_vec = torch.cat([slot.read() for slot in slots])
        instr_vec = model.embedder(instr)
        result = model(instr_vec, slot_vec)

        if debug:
            print(instr)
            pprint(dict(vars(SimulatorResult(*[x.data for x in result]))))
            print()

        wait_time = wait_time + result.wait_time[0]
        for slot in slots:
            slot.step(result.wait_time)

        for i in range(NUM_SLOTS):
            frac = result.write_head[i]
            overfill_loss = slots[i].mutate(frac * result.write_state, frac * result.write_time)
            schedule_loss = schedule_loss + frac * overfill_loss
            schedule_loss = schedule_loss + (frac > 1e-2).float() # l0 loss

    remaining_time = torch.max(torch.cat([slot.remaining_time for slot in slots]))
    total_time = wait_time + remaining_time
    wrongness_loss = F.mse_loss(total_time, torch.tensor(actual))
    loss = schedule_loss + 10 * wrongness_loss
    return ModelRunResult(total_time, loss, slots)

class Trainer(object):
    def __init__(self, model, train_data, name, save_freq=600):
        # type: (nn.Module, List[dt.DataItem], str, float) -> None
        self.model = model
        self.train_data = train_data
        self.name = name
        self.save_freq = save_freq
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
        self.last_save_time = 0 # type: float
        self.err_ema = 1

    def save_model(self):
        # type: () -> None
        fname = '{}_{}'.format(self.name, time.time())
        fpath =  os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch', 'saved', fname)
        torch.save(self.model.state_dict(), fpath)

    def load_latest(self):
        # type: () -> None
        raise NotImplementedError()

    def step_sgd(self, debug=False):
        # type: (bool) -> float
        scale = 100

        if time.time() - self.last_save_time > self.save_freq:
            self.save_model()
            self.last_save_time = time.time()

        datum = random.choice(self.train_data)

        self.optimizer.zero_grad()
        result = run_on_data(self.model, datum.block, datum.y / scale, debug=debug)
        result.loss.backward()
        self.optimizer.step()

        pred = result.prediction * scale

        err = 2 * abs((datum.y - pred) / (datum.y + pred))
        self.err_ema = 0.99 * self.err_ema + 0.01 * err

        return err

    def loop_sgd(self):
        # type: () -> None
        i = 0
        while True:
            err = self.step_sgd()
            i += 1
            if i > 100:
                print('err_ema: {:.2f}, err: {:.2f}'.format(self.err_ema, err), end='\r')
                i = 0

    def debug_sgd(self):
        # type: () -> None
        err = self.step_sgd(debug=True)
