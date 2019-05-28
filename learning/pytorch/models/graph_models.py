import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

from enum import Enum, unique
import torch
import torch.nn as nn
import torch.nn.functional as F
import common_libs.utilities as ut
import data.data_cost as dt
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple
from . import model_utils

class AbstractGraphModule(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_classes):
        # type: (int, int, int) -> None
        super(AbstractGraphModule, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

    def set_learnable_embedding(self, mode, dictsize, seed = None):
        # type: (str, int, Optional[int]) -> None

        self.mode = mode

        if mode != 'learnt':
            embedding = nn.Embedding(dictsize, self.embedding_size)

        if mode == 'none':
            print 'learn embeddings form scratch...'
            initrange = 0.5 / self.embedding_size
            embedding.weight.data.uniform_(-initrange, initrange)
            self.final_embeddings = embedding
        elif mode == 'seed':
            print 'seed by word2vec vectors....'
            embedding.weight.data = torch.FloatTensor(seed)
            self.final_embeddings = embedding
        elif mode == 'learnt':
            print 'using learnt word2vec embeddings...'
            self.final_embeddings = seed
        else:
            print 'embedding not selected...'
            exit()

    def dump_shared_params(self):
        # type: () -> Dict[str, Any]
        return model_utils.dump_shared_params(self)

    def load_shared_params(self, params):
        # type: (Dict[str, Any]) -> None
        model_utils.load_shared_params(self, params)

    def load_checkpoint_file(self, fname):
        self.load_state_dict(torch.load(fname)['model'])

    def load_state_dict(self, state_dict):
        model_dict = self.state_dict()
        new_model_dict = {k: v for (k, v) in state_dict.items() if k in model_dict}
        model_dict.update(new_model_dict)
        super(AbstractGraphModule, self).load_state_dict(model_dict)

    def init_hidden(self):
        # type: () -> Tuple[nn.Parameter, nn.Parameter]

        return (
            nn.Parameter(torch.zeros(1, 1, self.hidden_size, requires_grad=True)),
            nn.Parameter(torch.zeros(1, 1, self.hidden_size, requires_grad=True)),
        )

    def remove_refs(self, item):
        # type: (dt.DataItem) -> None
        pass

@unique
class ReductionType(Enum):
    MAX = 0
    ADD = 1
    MEAN = 2
    ATTENTION = 3

@unique
class NonlinearityType(Enum):
    RELU = 0
    SIGMOID = 1
    TANH = 2

class GraphNN(AbstractGraphModule):

    def __init__(self, embedding_size, hidden_size, num_classes, use_residual=True, linear_embed=False, use_dag_rnn=True, reduction=ReductionType.MAX, nonlinear_width=128, nonlinear_type=NonlinearityType.RELU, nonlinear_before_max=False):
        # type: (int, int, int, bool, bool, bool, ReductionType, int, NonlinearityType, bool) -> None
        super(GraphNN, self).__init__(embedding_size, hidden_size, num_classes)

        assert use_residual or use_dag_rnn, 'Must use some type of predictor'

        self.use_residual = use_residual
        self.linear_embed = linear_embed
        self.use_dag_rnn = use_dag_rnn

        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

        # linear weight for instruction embedding
        self.opcode_lin = nn.Linear(self.embedding_size, self.hidden_size)
        self.src_lin = nn.Linear(self.embedding_size, self.hidden_size)
        self.dst_lin = nn.Linear(self.embedding_size, self.hidden_size)
        # for sequential model
        self.opcode_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)
        self.src_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)
        self.dst_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)

        #linear layer for final regression result
        self.linear = nn.Linear(self.hidden_size,self.num_classes)

        self.nonlinear_1 = nn.Linear(self.hidden_size, nonlinear_width)
        self.nonlinear_2 = nn.Linear(nonlinear_width, self.num_classes)

        #lstm - for sequential model
        self.lstm_token_seq = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins_seq = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear_seq = nn.Linear(self.hidden_size, self.num_classes)

        self.reduction_typ = reduction
        self.attention_1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.attention_2 = nn.Linear(self.hidden_size // 2, 1)

        self.nonlinear_premax_1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.nonlinear_premax_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.nonlinear_seq_1 = nn.Linear(self.hidden_size, nonlinear_width)
        self.nonlinear_seq_2 = nn.Linear(nonlinear_width, self.num_classes)

        self.use_nonlinear = nonlinear_type is not None

        if nonlinear_type == NonlinearityType.RELU:
            self.final_nonlinearity = torch.relu
        elif nonlinear_type == NonlinearityType.SIGMOID:
            self.final_nonlinearity = torch.sigmoid
        elif nonlinear_type == NonlinearityType.TANH:
            self.final_nonlinearity = torch.tanh

        self.nonlinear_before_max = nonlinear_before_max

    def reduction(self, items):
        # type: (List[torch.tensor]) -> torch.tensor
        if len(items) == 0:
            return self.init_hidden()[0]
        elif len(items) == 1:
            return items[0]

        def binary_reduction(reduction):
            # type: (Callable[[torch.tensor, torch.tensor], torch.tensor]) -> torch.tensor
            final = items[0]
            for item in items[1:]:
                final = reduction(final, item)
            return final

        stacked_items = torch.stack(items)

        if self.reduction_typ == ReductionType.MAX:
            return binary_reduction(torch.max)
        elif self.reduction_typ == ReductionType.ADD:
            return binary_reduction(torch.add)
        elif self.reduction_typ == ReductionType.MEAN:
            return binary_reduction(torch.add) / len(items)
        elif self.reduction_typ == ReductionType.ATTENTION:
            preds = torch.stack([self.attention_2(torch.relu(self.attention_1(item))) for item in items])
            probs = F.softmax(preds, dim=0)
            print('{}, {}, {}'.format(
                probs.shape,
                stacked_items.shape,
                stacked_items * probs
            ))
            return (stacked_items * probs).sum(dim=0)
        else:
            raise ValueError()

    def remove_refs(self, item):
        # type: (dt.DataItem) -> None

       for instr in item.block.instrs:
            if instr.lstm != None:
                del instr.lstm
            if instr.hidden != None:
                del instr.hidden
            instr.lstm = None
            instr.hidden = None
            instr.tokens = None

    def init_bblstm(self, item):
        # type: (dt.DataItem) -> None

        self.remove_refs(item)
        for i, instr in enumerate(item.block.instrs):
            tokens = item.x[i]
            if self.mode == 'learnt':
                instr.tokens = [self.final_embeddings[token] for token in tokens]
            else:
                instr.tokens = self.final_embeddings(torch.LongTensor(tokens))

    def create_graphlstm(self, block):
        # type: (ut.BasicBlock) -> torch.tensor

        leaves = block.find_leaves()

        leaf_hidden = []
        for leaf in leaves:
            hidden = self.create_graphlstm_rec(leaf)
            leaf_hidden.append(hidden[0].squeeze())

        if self.nonlinear_before_max:
            leaf_hidden = [
                self.nonlinear_premax_2(torch.relu(self.nonlinear_premax_1(h)))
                for h in leaf_hidden
            ]

        return self.reduction(leaf_hidden)

    def get_instruction_embedding_linear(self, instr, seq_model):
        # type: (ut.Instruction, bool) -> torch.tensor

        if seq_model:
            opcode_lin = self.opcode_lin_seq
            src_lin = self.src_lin_seq
            dst_lin = self.dst_lin_seq
        else:
            opcode_lin = self.opcode_lin
            src_lin = self.src_lin
            dst_lin = self.dst_lin

        opc_embed = instr.tokens[0]
        src_embed = instr.tokens[2:2+len(instr.srcs)]
        dst_embed = instr.tokens[-1-len(instr.dsts):-1]

        opc_hidden = opcode_lin(opc_embed)

        src_hidden = torch.zeros(self.embedding_size)
        for s in src_embed:
            src_hidden = torch.max(F.relu(src_lin(s)))

        dst_hidden = torch.zeros(self.embedding_size)
        for d in dst_embed:
            dst_hidden = torch.max(F.relu(dst_lin(d)))

        return (opc_hidden + src_hidden + dst_hidden).unsqueeze(0).unsqueeze(0)


    def get_instruction_embedding_lstm(self, instr, seq_model):
        # type: (ut.Instruction, bool) -> torch.tensor
        if seq_model:
            lstm = self.lstm_token_seq
        else:
            lstm = self.lstm_token

        _, hidden = lstm(instr.tokens.unsqueeze(1), self.init_hidden())
        return hidden[0]

    def get_instruction_embedding(self, instr, seq_model):
        # type: (ut.Instruction, bool) -> torch.tensor
        if self.linear_embed:
            return self.get_instruction_embedding_linear(instr, seq_model)
        else:
            return self.get_instruction_embedding_lstm(instr, seq_model)

    def create_graphlstm_rec(self, instr):
        # type: (ut.Instruction) -> torch.tensor

        if instr.hidden != None:
            return instr.hidden

        parent_hidden = [self.create_graphlstm_rec(parent) for parent in instr.parents]

        if len(parent_hidden) > 0:
            hs, cs = list(zip(*parent_hidden))
            in_hidden_ins = (self.reduction(hs), self.reduction(cs))
        else:
            in_hidden_ins = self.init_hidden()

        ins_embed = self.get_instruction_embedding(instr, False)

        out_ins, hidden_ins = self.lstm_ins(ins_embed, in_hidden_ins)
        instr.hidden = hidden_ins

        return instr.hidden

    def create_residual_lstm(self, block):
        # type: (ut.BasicBlock) -> torch.tensor

        ins_embeds = autograd.Variable(torch.zeros(len(block.instrs),self.embedding_size))
        for i, ins in enumerate(block.instrs):
            ins_embeds[i] = self.get_instruction_embedding(ins, True).squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        _, hidden_ins = self.lstm_ins_seq(ins_embeds_lstm, self.init_hidden())

        seq_ret = hidden_ins[0].squeeze()

        return seq_ret

    def forward(self, item):
        # type: (dt.DataItem) -> torch.tensor

        self.init_bblstm(item)

        final_pred = torch.zeros(self.num_classes).squeeze()

        if self.use_dag_rnn:
            graph = self.create_graphlstm(item.block)
            if self.use_nonlinear and not self.nonlinear_before_max:
                final_pred += self.nonlinear_2(self.final_nonlinearity(self.nonlinear_1(graph))).squeeze()
            else:
                final_pred += self.linear(graph).squeeze()

        if self.use_residual:
            sequential = self.create_residual_lstm(item.block)
            if self.use_nonlinear:
                final_pred += self.nonlinear_seq_2(self.final_nonlinearity(self.nonlinear_seq_1(sequential))).squeeze()
            else:
                final_pred += self.linear(sequential).squeeze()

        return final_pred.squeeze()

@unique
class RnnHierarchyType(Enum):
    NONE = 0
    DENSE =  1
    MULTISCALE = 2
    LINEAR_MODEL = 3
    MOP_MODEL = 4

@unique
class RnnType(Enum):
    RNN = 0
    LSTM = 1
    GRU = 2

RnnParameters = NamedTuple('RnnParameters', [
    ('embedding_size', int),
    ('hidden_size', int),
    ('num_classes', int),
    ('connect_tokens', bool),
    ('skip_connections', bool),
    ('learn_init', bool),
    ('hierarchy_type', RnnHierarchyType),
    ('rnn_type', RnnType),
])


class RNN(AbstractGraphModule):

    def __init__(self, params):
        # type: (RnnParameters) -> None
        super(RNN, self).__init__(params.embedding_size, params.hidden_size, params.num_classes)

        self.params = params

        if params.rnn_type == RnnType.RNN:
            self.token_rnn = nn.RNN(self.embedding_size, self.hidden_size)
            self.instr_rnn = nn.RNN(self.hidden_size, self.hidden_size)
        elif params.rnn_type == RnnType.LSTM:
            self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size)
            self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        elif params.rnn_type == RnnType.GRU:
            self.token_rnn = nn.GRU(self.embedding_size, self.hidden_size)
            self.instr_rnn = nn.GRU(self.hidden_size, self.hidden_size)
        else:
            raise ValueError('Unknown RNN type {}'.format(params.rnn_type))

        self._token_init = self.rnn_init_hidden()
        self._instr_init = self.rnn_init_hidden()

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def rnn_init_hidden(self):
        # type: () -> Union[Tuple[nn.Parameter, nn.Parameter], nn.Parameter]

        hidden = self.init_hidden()

        # for h in hidden:
        #     torch.nn.init.kaiming_uniform_(h)

        if self.params.rnn_type == RnnType.LSTM:
            return hidden
        else:
            return hidden[0]

    def get_token_init(self):
        # type: () -> torch.tensor
        if self.params.learn_init:
            return self._token_init
        else:
            return self.rnn_init_hidden()

    def get_instr_init(self):
        # type: () -> torch.tensor
        if self.params.learn_init:
            return self._instr_init
        else:
            return self.rnn_init_hidden()

    def pred_of_instr_chain(self, instr_chain):
        # type: (torch.tensor) -> torch.tensor
        _, final_state_packed = self.instr_rnn(instr_chain, self.get_instr_init())
        if self.params.rnn_type == RnnType.LSTM:
            final_state = final_state_packed[0]
        else:
            final_state = final_state_packed

        return self.linear(final_state.squeeze()).squeeze()


    def forward(self, item):
        # type: (dt.DataItem) -> torch.tensor

        token_state = self.get_token_init()

        token_output_map = {} # type: Dict[ut.Instruction, torch.tensor]
        token_state_map = {} # type: Dict[ut.Instruction, torch.tensor]

        for instr, token_inputs in zip(item.block.instrs, item.x):
            if not self.params.connect_tokens:
                token_state = self.get_token_init()

            if self.params.skip_connections and self.params.hierarchy_type == RnnHierarchyType.NONE:
                for parent in instr.parents:
                    parent_state = token_state_map[parent]

                    if self.params.rnn_type == RnnType.LSTM:
                        token_state = (
                            token_state[0] + parent_state[0],
                            token_state[1] + parent_state[1],
                        )
                    else:
                        token_state = token_state + parent_state

            tokens = self.final_embeddings(torch.LongTensor(token_inputs)).unsqueeze(1)
            output, state = self.token_rnn(tokens, token_state)
            token_output_map[instr] = output
            token_state_map[instr] = state

        if self.params.hierarchy_type == RnnHierarchyType.NONE:
            final_state_packed = token_state_map[item.block.instrs[-1]]

            if self.params.rnn_type == RnnType.LSTM:
                final_state = final_state_packed[0]
            else:
                final_state = final_state_packed
            return self.linear(final_state.squeeze()).squeeze()

        instr_chain = torch.stack([token_output_map[instr][-1] for instr in item.block.instrs])

        if self.params.hierarchy_type == RnnHierarchyType.DENSE:
            instr_chain = torch.stack([state for instr in item.block.instrs for state in token_output_map[instr]])
        elif self.params.hierarchy_type == RnnHierarchyType.LINEAR_MODEL:
            return sum(
                self.linear(st).squeeze()
                for st in instr_chain
            )
        elif self.params.hierarchy_type == RnnHierarchyType.MOP_MODEL:
            preds = torch.stack([
                self.pred_of_instr_chain(torch.stack([token_output_map[instr][-1] for instr in instrs]))
                for instrs in item.block.paths_of_block()
            ])
            return torch.max(preds)

        return self.pred_of_instr_chain(instr_chain)

class Fasthemal(AbstractGraphModule):
    def __init__(self, embedding_size, hidden_size, num_classes):
        # type: (int, int, int) -> None
        super(Fasthemal, self).__init__(embedding_size, hidden_size, num_classes)
        self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size)
        self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, item):
        # type: (dt.DataItem) -> torch.tensor

        embeds = []

        for token_inputs in item.x:
            tokens = self.final_embeddings(torch.LongTensor(token_inputs)).unsqueeze(1)
            _, (token_state, _) = self.token_rnn(tokens)
            embeds.append(token_state.squeeze(1))

        z = torch.stack(embeds)
        _, instr_state = self.instr_rnn(z)

        return self.linear(instr_state[0].squeeze()).squeeze()
