from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import common_libs.utilities as ut
import data.data_cost as dt
import functools
import itertools
from pprint import pprint
import random
import re
import sparsemax
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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
        perc = additional_time / (additional_time + self.remaining_time)
        self.state = new_state * perc + self.state * (1 - perc)
        old_time = self.remaining_time
        self.remaining_time = self.remaining_time + additional_time
        return old_time[0]

    def step(self, time):
        # type: (torch.tensor) -> None
        self.remaining_time = torch.clamp(self.remaining_time - time, min=0)

    def read(self):
        # type: () -> torch.tensor
        state = self.state * torch.clamp(self.remaining_time * 10, 0, 1)
        return torch.cat([self.remaining_time, state])

class NeuralProcessorSimulator(nn.Module):
    def __init__(self):
        # type: () -> None
        super(NeuralProcessorSimulator, self).__init__()
        latent_width = 256
        sym_dict, _ = ut.get_sym_dict()
        self._embed_width = 128
        self._n_embeddings = len(sym_dict)
        self._max_n_srcs = 3
        self._max_n_dsts = 3
        self.embedder = torch.nn.Embedding(self._n_embeddings, self._embed_width)
        full_embed_width = self._embed_width * (1 + self._max_n_srcs + self._max_n_dsts)
        self.instr_vec_emb = nn.Linear(full_embed_width, latent_width // 2)
        self.slot_vec_emb = nn.Linear((1+SLOT_WIDTH)*NUM_SLOTS, latent_width // 2)
        self.latent_layer = nn.Linear(latent_width, latent_width)
        self.wait_time_out = nn.Linear(latent_width, 1)
        self.write_head_out = nn.Linear(latent_width, NUM_SLOTS)
        self.write_state_out = nn.Linear(latent_width, SLOT_WIDTH)
        self.write_time_out = nn.Linear(latent_width, 1)

        self.write_head_sparsemax = sparsemax.Sparsemax(dim=0)

    def _get_emb_list(self, arr, length):
        # type: (List[int], int) -> List[torch.tensor]
        assert len(arr) <= length
        clamp = lambda x: x if x < self._n_embeddings else self._n_embeddings - 1
        real = [self.embedder(torch.tensor(clamp(val))) for val in arr]
        zeros = [torch.zeros(self._embed_width) for _ in range(length - len(arr))]
        return real + zeros

    def embed(self, instr):
        # type: (ut.Instruction) -> torch.tensor
        opc = self.embedder(torch.tensor(instr.opcode))
        srcs = self._get_emb_list(instr.srcs, self._max_n_srcs)
        dsts = self._get_emb_list(instr.dsts, self._max_n_dsts)
        return torch.cat([opc] + srcs + dsts)

    def forward(self, instr_vec, slot_vec):
        # type: (torch.tensor, torch.tensor) -> SimulatorResult
        instr_vec = self.instr_vec_emb(instr_vec)
        slot_vec = self.slot_vec_emb(slot_vec)
        concat = F.relu(torch.cat([instr_vec, slot_vec]))
        latent = F.relu(self.latent_layer(concat))

        wait_time = self.wait_time_out(latent).abs()
        write_head = self.write_head_sparsemax(self.write_head_out(latent))
        write_state = self.write_state_out(latent)
        write_time = self.write_time_out(latent).abs()

        return SimulatorResult(
            wait_time=wait_time,
            write_head=write_head,
            write_state=write_state,
            write_time=write_time,
        )

def run_on_data(model, block, actual, debug=False):
    # type: (nn.Module, ut.BasicBlock, float, bool) -> ModelRunResult
    slots = [Slot() for _ in range(NUM_SLOTS)]
    schedule_loss = torch.tensor(0.)
    wait_time = torch.tensor(0.)

    for instr in block.instrs:
        slot_vec = torch.cat([slot.read() for slot in slots])
        instr_vec = model.embed(instr)
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
            schedule_loss = schedule_loss + frac # l1 loss

    remaining_time = torch.max(torch.cat([slot.remaining_time for slot in slots]))
    total_time = wait_time + remaining_time
    wrongness_loss = F.mse_loss(total_time, torch.tensor(actual))
    loss = schedule_loss + 100 * wrongness_loss
    return ModelRunResult(total_time, loss, slots)

class Trainer(object):
    def __init__(self, model, train_data, name, save_freq=600):
        # type: (nn.Module, List[dt.DataItem], str, float) -> None
        self.model = model
        self.train_data = train_data
        self.name = name
        self.save_freq = save_freq
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)
        self.last_save_time = 0 # type: float
        self.err_ema = 1

    def save_model(self):
        # type: () -> None
        fname = '{}_{}'.format(self.name, time.time())
        fpath =  os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch', 'saved', fname)
        torch.save(self.model.state_dict(), fpath)

    def load_model(self, fname):
        # type: (str) -> None
        self.model.load_state_dict(torch.load(fname))

    def load_latest(self):
        # type: () -> None
        pat = re.compile(r'{}_([\d\.]+)$'.format(re.escape(self.name)))
        dname = os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch', 'saved')
        fnames = filter(bool, map(pat.match, os.listdir(dname)))
        latest_fname = max(fnames, key=lambda m: float(m.group(1))).group(0)

        self.load_model(os.path.join(dname, latest_fname))

    def step_sgd(self, debug=False):
        # type: (bool) -> float
        scale = 268

        if time.time() - self.last_save_time > self.save_freq:
            self.save_model()
            self.last_save_time = time.time()

        datum = random.choice(self.train_data)

        if datum.y < 5 or datum.y > 1e6:
            return 1

        self.optimizer.zero_grad()
        result = run_on_data(self.model, datum.block, datum.y / scale, debug=debug)
        result.loss.backward()
        self.optimizer.step()

        pred = result.prediction.item() * scale

        err = abs((datum.y - pred) / datum.y)
        ema_exp = 0.999
        self.err_ema = ema_exp * self.err_ema + (1 - ema_exp) * err

        return err

    def loop_sgd(self):
        # type: () -> None
        format_desc = desc='Error EMA: {:.2f}'.format
        pbar = tqdm(desc=format_desc(1, 0))
        for iter_idx in itertools.count(1):
            err = self.step_sgd()
            pbar.set_description(format_desc(self.err_ema, iter_idx), refresh=False)
            pbar.update(1)

    def debug_sgd(self):
        # type: () -> None
        err = self.step_sgd(debug=True)
