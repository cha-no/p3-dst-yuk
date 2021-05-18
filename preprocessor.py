import numpy as np
import torch

from data_utils import (
    DSTPreprocessor,
    OpenVocabDSTFeature,
    convert_state_dict,
    _truncate_seq_pair,
    OntologyDSTFeature,
    SOMDSTFeature,
    OP_SET,
    EXPERIMENT_DOMAINS,
)

flatten = lambda x: [i for s in x for i in s]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
        word_dropout=0.0,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        # self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2}
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout

    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)

        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:]

        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
                continue

            # if self.id2gating[gate] == "dontcare":
            #     recovered.append("%s-%s" % (slot, "dontcare"))
            #     continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]

        if self.word_dropout > 0.0:
            input_ids = []
            for b in batch:
                drop_mask = (np.array(self.src_tokenizer.get_special_tokens_mask(b.input_id, already_has_special_tokens=True)) == 0).astype(int)
                word_drop = np.random.binomial(drop_mask, self.word_dropout)
                input_id = [
                    token_id if word_drop[i] == 0 else self.src_tokenizer.unk_token_id
                    for i, token_id in enumerate(b.input_id)
                ]
                input_ids.append(input_id) 
            input_ids = torch.LongTensor(
                self.pad_ids([b for b in input_ids], self.src_tokenizer.pad_token_id)
            )
        else:
            input_ids = torch.LongTensor(
                self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
            )

        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids


class SUMBTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=64,
        max_turn_length=14,
        word_dropout=0.0,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq_length = max_seq_length
        self.max_turn_length = max_turn_length
        self.word_dropout = word_dropout

    def _convert_example_to_feature(self, example):
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        num_turn = None
        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2
            uttrs = []
            for segment_idx, uttr in enumerate(turn.current_turn):
                token = self.src_tokenizer.encode(uttr, add_special_tokens=False)
                uttrs.append(token)

            _truncate_seq_pair(uttrs[0], uttrs[1], self.max_seq_length - 3)
            tokens = (
                [self.src_tokenizer.cls_token_id]
                + uttrs[0]
                + [self.src_tokenizer.sep_token_id]
                + uttrs[1]
                + [self.src_tokenizer.sep_token_id]
            )
            token_type = [0] * (len(uttrs[0]) + 2) + [1] * (len(uttrs[1]) + 1)
            if len(tokens) < self.max_seq_length:
                gap = self.max_seq_length - len(tokens)
                tokens.extend([self.src_tokenizer.pad_token_id] * gap)
                token_type.extend([0] * gap)
            turns.append(tokens)
            token_types.append(token_type)
            label = []
            if turn.label:
                slot_dict = convert_state_dict(turn.label)
            else:
                slot_dict = {}
            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")
                if value in self.ontology[slot_type]:
                    label_idx = self.ontology[slot_type].index(value)
                else:
                    label_idx = self.ontology[slot_type].index("none")
                                
                label.append(label_idx)
            labels.append(label)
        num_turn = len(turns)
        if len(turns) < self.max_turn_length:
            gap = self.max_turn_length - len(turns)
            for _ in range(gap):
                dummy_turn = [self.src_tokenizer.pad_token_id] * self.max_seq_length
                turns.append(dummy_turn)
                token_types.append(dummy_turn)
                dummy_label = [-1] * len(self.slot_meta)
                labels.append(dummy_label)
        return OntologyDSTFeature(
            guid=guid,
            input_ids=turns,
            segment_ids=token_types,
            num_turn=num_turn,
            target_ids=labels,
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, pred_slots, num_turn):
        states = []
        for pred_slot in pred_slots[:num_turn]:
            state = []
            for s, p in zip(self.slot_meta, pred_slot):
                v = self.ontology[s][p]
                if v != 'none':
                    state.append(f"{s}-{v}")
            states.append(state)
        return states

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]

        if self.word_dropout > 0.0:
            input_ids = []
            for b in batch:
                drop_mask = (np.array(self.src_tokenizer.get_special_tokens_mask(b.input_id, already_has_special_tokens=True)) == 0).astype(int)
                word_drop = np.random.binomial(drop_mask, self.word_dropout)
                input_id = [
                    token_id if word_drop[i] == 0 else self.src_tokenizer.unk_token_id
                    for i, token_id in enumerate(b.input_id)
                ]
                input_ids.append(input_id) 
            input_ids = torch.LongTensor(
                self.pad_ids([b for b in input_ids], self.src_tokenizer.pad_token_id)
            )
        else:
            input_ids = torch.LongTensor(
                self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
            )

        # input_ids = torch.LongTensor([b.input_ids for b in batch])
        segment_ids = torch.LongTensor([b.segment_ids for b in batch])
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)
        target_ids = torch.LongTensor([b.target_ids for b in batch])
        num_turns = [b.num_turn for b in batch]
        return input_ids, segment_ids, input_masks, target_ids, num_turns, guids

def tokenize_ontology(ontology, tokenizer, max_seq_length = 12):
    slot_types, slot_values = [], []
    
    for (slot, values) in ontology.items():
        token = tokenizer.encode(slot)
        if len(token) < max_seq_length:
            gap = max_seq_length - len(token)
            token.extend([tokenizer.pad_token_id] * gap)
        slot_types.append(token)
        
        slot_value = []
        for value in values:
            token = tokenizer.encode(value)
            if len(token) < max_seq_length:
                gap = max_seq_length - len(token)
                token.extend([tokenizer.pad_token_id] * gap)
            slot_value.append(token)
        slot_values.append(torch.LongTensor(slot_value))
    return torch.LongTensor(slot_types), slot_values
        
class SOMDSTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        n_history = 1,
        slot_token = '[SLOT]',
        separate_token = ';',
        ontology=None,
        max_seq_length=512,
        word_dropout=0.0,
        op_code='4'
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.n_history = n_history
        self.slot_token = slot_token
        self.separate_token = separate_token
        self.ontology = ontology
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout
        self.op2id = OP_SET[op_code]
        self.op_ids = None
        self.generate_ids = None        
        
        special_tokens_dict = {'additional_special_tokens': [slot_token, '[NULL]', '[EOS]']}
        num_added_toks = self.src_tokenizer.add_special_tokens(special_tokens_dict)
        
    def _convert_example_to_feature(self, example):
        max_seq_length = self.max_seq_length
        state = []
        
        for s in self.slot_meta:
            state.append(self.slot_token)
            k = s.split('-')
            v = example.last_dialog_state.get(s)
            if v is not None:
                k.extend(['-', v])
                t = self.src_tokenizer.tokenize(' '.join(k))
            else:
                t = self.src_tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)
        
        dialog_history = self.separate_token.join(example.context_turns[-2 * self.n_history:])
        turn_utter = self.separate_token.join(example.current_turn)
        
        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = self.src_tokenizer.tokenize(dialog_history)
        diag_2 = self.src_tokenizer.tokenize(turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2
        # word dropout
        if self.word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), self.word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]

        input_ = diag + state
        segment = segment + [1] * len(state)
        self.input_ = input_

        self.segment_id = segment
        slot_position = []
        for i, t in enumerate(self.input_):
            if t == self.slot_token:
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = self.src_tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.domain_id = domain2id[example.turn_domain]
        if example.op_labels:
            self.op_ids = [self.op2id[a] for a in example.op_labels]
        if example.generate_y:
            self.generate_ids = [self.src_tokenizer.convert_tokens_to_ids(y) for y in example.generate_y]
        else:
            self.generate_ids = []
        return SOMDSTFeature(
            example.guid, self.input_id, self.segment_id, self.input_mask,
            self.slot_position, self.op_ids, self.domain_id, self.generate_ids, example.last_dialog_state, example.is_last_turn
        )


    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def check_testdata(self, batch):
        for b in batch:
            if b.op_ids is not None:
                return True
            else:
                return False
        
    def collate_fn(self, batch):
        input_ids = torch.tensor([b.input_id for b in batch], dtype=torch.long)
        input_mask = torch.tensor([b.input_mask for b in batch], dtype=torch.long)
        segment_ids = torch.tensor([b.segment_id for b in batch], dtype=torch.long)
        state_position_ids = torch.tensor([b.slot_position for b in batch], dtype=torch.long)
        domain_ids = torch.tensor([b.domain_id for b in batch], dtype=torch.long)
        if self.check_testdata(batch):
            op_ids = torch.tensor([b.op_ids for b in batch], dtype=torch.long)
            gen_ids = [b.generate_ids for b in batch]
            max_update = max([len(b) for b in gen_ids])
            max_value = max([len(b) for b in flatten(gen_ids)]) if flatten(gen_ids) else 0
            for bid, b in enumerate(gen_ids):
                n_update = len(b)
                for idx, v in enumerate(b):
                    b[idx] = v + [0] * (max_value - len(v))
                gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
            gen_ids = torch.tensor(gen_ids, dtype=torch.long)
        else:
            op_ids, gen_ids, max_update, max_value = 0, 0, 0, 0
        
        return input_ids, input_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, max_value, max_update