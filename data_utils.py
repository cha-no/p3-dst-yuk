import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import os
import json
import random
from collections import defaultdict
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

EXPERIMENT_DOMAINS = ['관광', '숙소', '식당', '지하철', '택시']
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}
OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}


@dataclass
class OntologyDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    num_turn: int
    target_ids: Optional[List[int]]


@dataclass
class OpenVocabDSTFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]


@dataclass
class SOMDSTFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    input_mask: List[int]
    slot_position: Optional[List[int]] = None
    op_ids: Optional[List[int]] = None
    domain_id: Optional[List[int]] = None
    generate_ids: Optional[List[int]] = None
    last_dialog_state: Optional[Dict[str, str]] = None
    is_last_turn: Optional[bool] = None


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


def load_dataset(dataset_path, dev_split=0.1):
    data = json.load(open(dataset_path))
    num_data = len(data)
    num_dev = int(num_data * dev_split)
    if not num_dev:
        return data, []  # no dev dataset

    dom_mapper = defaultdict(list)
    for d in data:
        dom_mapper[len(d["domains"])].append(d["dialogue_idx"])

    num_per_domain_trainsition = int(num_dev / 3)
    dev_idx = []
    for v in dom_mapper.values():
        idx = random.sample(v, num_per_domain_trainsition)
        dev_idx.extend(idx)

    train_data, dev_data = [], []
    for d in data:
        if d["dialogue_idx"] in dev_idx:
            dev_data.append(d)
        else:
            train_data.append(d)

    dev_labels = {}
    for dialogue in dev_data:
        d_idx = 0
        guid = dialogue["dialogue_idx"]
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            state = turn.pop("state")

            guid_t = f"{guid}-{d_idx}"
            d_idx += 1

            dev_labels[guid_t] = state

    return train_data, dev_data, dev_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def split_slot(dom_slot_value, get_domain_slot=False):
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def build_slot_meta(data):
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue

            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def convert_state_dict(state):
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic


# @dataclass
# class DSTInputExample:
#     guid: str
#     context_turns: List[str]
#     current_turn: List[str]
#     label: Optional[List[str]] = None

#     def to_dict(self):
#         return dataclasses.asdict(self)

#     def to_json_string(self):
#         """Serializes this instance to a JSON string."""
#         return json.dumps(self.to_dict(), indent=2) + "\n"

@dataclass
class DSTInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    turn_domain: Optional[str]
    last_dialog_state: Optional[Dict[str, str]]
    op_labels: Optional[List[str]]
    generate_y: Optional[List[List[str]]]
    label: Optional[List[str]] = None
    is_last_turn: Optional[bool] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

def get_turn_dialog_state(state):
    turn_dialog_state = {}
    for s in state:
        domain, slot, value = s.split('-')
        turn_dialog_state['-'.join([domain, slot])] = value
    return turn_dialog_state


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_examples_from_dialogue(dialogue, slot_meta, labels = None, user_first=False, tokenizer = None):
    guid = dialogue["dialogue_idx"]
    turn_domain = get_turn_domain(guid)
    examples = []
    history = []
    d_idx = 0
    op_labels = ['carryover'] * len(slot_meta) if slot_meta else None
    generate_y = None
    last_dialog_state = {}
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")
        if state is None:
            if labels:
                state = labels[guid + f'-{d_idx}']
            else:
                state = None
        
        context = deepcopy(history)
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]

        if state and slot_meta:
            turn_dialog_state = get_turn_dialog_state(state)
            op_labels, generate_y = make_turn_label(slot_meta, last_dialog_state,
                                                                turn_dialog_state,
                                                                tokenizer, op_code='4')

        if (idx + 2) == len(dialogue["dialogue"]):
            is_last_turn = True
        else:
            is_last_turn = False

        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
                turn_domain=turn_domain,
                last_dialog_state=last_dialog_state,
                op_labels=op_labels,
                generate_y=generate_y,
                is_last_turn=is_last_turn
            )
        )
        if state and slot_meta:
            last_dialog_state = turn_dialog_state
        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1
    return examples

# def get_examples_from_dialogue(dialogue, user_first=False):
#     guid = dialogue["dialogue_idx"]
#     examples = []
#     history = []
#     d_idx = 0
#     for idx, turn in enumerate(dialogue["dialogue"]):
#         if turn["role"] != "user":
#             continue

#         if idx:
#             sys_utter = dialogue["dialogue"][idx - 1]["text"]
#         else:
#             sys_utter = ""

#         user_utter = turn["text"]
#         state = turn.get("state")
#         context = deepcopy(history)
#         if user_first:
#             current_turn = [user_utter, sys_utter]
#         else:
#             current_turn = [sys_utter, user_utter]
#         examples.append(
#             DSTInputExample(
#                 guid=f"{guid}-{d_idx}",
#                 context_turns=context,
#                 current_turn=current_turn,
#                 label=state,
#             )
#         )
#         history.append(sys_utter)
#         history.append(user_utter)
#         d_idx += 1
#     return examples


def get_examples_from_dialogues(data, slot_meta = None, labels = None, user_first=False, dialogue_level=False, tokenizer = None):
    examples = []
    for d in tqdm(data):
        example = get_examples_from_dialogue(d, slot_meta, labels, user_first=user_first, tokenizer = tokenizer)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples

# def get_examples_from_dialogues(data, user_first=False, dialogue_level=False):
#     examples = []
#     for d in tqdm(data):
#         example = get_examples_from_dialogue(d, user_first=user_first)
#         if dialogue_level:
#             examples.append(example)
#         else:
#             examples.extend(example)
#     return examples

def get_turn_domain(guid):
    _, gid = guid.split(':')
    return gid.split('_')[0]


def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    keys = list(turn_dialog_state.keys())
    for k in keys:
        v = turn_dialog_state[k]
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
            elif vv == v:
                op_labels[idx] = 'carryover'
        except ValueError:
            continue

    for k, v in last_dialog_state.items():
        vv = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([['[NULL]', '[EOS]'], idx])
        except ValueError:
            continue

    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y

class DSTPreprocessor:
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def _convert_example_to_feature(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def recover_state(self):
        raise NotImplementedError
