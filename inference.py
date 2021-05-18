import argparse
import os
import json

from typing import Any

from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

from data_utils import (WOSDataset, get_examples_from_dialogues)
from model import *
from preprocessor import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
op_code = '4'
op2id = OP_SET[op_code]

def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state

def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen
    return generated, last_dialog_state

def load_model(args : argparse.Namespace, config, tokenizer : BertTokenizer, ontology : dict, slot_meta : list, device : torch.device) -> nn.Module:
    """
    Model settings according to args.
    Args:
        args (argparse.Namespace).
        tokenizer (BertTokenizer): "dsksd/bert-ko-small-minimal".
        ontology (dict).
        slot_meta (list).
        device (torch.device).
    Return:
        Model (nn.Module)
    """
    if config.architecture == 'TRADE':
        # Slot Meta tokenizing for the decoder initial inputs
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model = TRADE(config, tokenized_slot_meta)
        ckpt = torch.load(os.path.join(args.model_dir, args.model_name), map_location="cpu")
        model.load_state_dict(ckpt)
    elif config.architecture == 'SUMBT':
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, config.max_label_length)
        num_labels = [len(s) for s in slot_values_ids]
        model = SUMBT(config, num_labels, device)
        ckpt = torch.load(os.path.join(args.model_dir, args.model_name), map_location="cpu")
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)
        model.load_state_dict(ckpt)
    elif config.architecture == 'SOMDST':
        model = SomDST(config, len(op2id), len(domain2id), op2id['update'])        
        ckpt = torch.load(os.path.join(args.model_dir, args.model_name), map_location="cpu")
        model.load_state_dict(ckpt)

    return model

def _inference(args : argparse.Namespace, model : nn.Module, batch : int, processor : Any, predictions : dict, n_gpu : int, device : torch.Tensor) -> dict:
    """
    Predict slot_value.
    Args:
        args (argparse.Namespace).
        model (nn.Module).
        batch (int).
        processor (Any).
        predictions (dict).
        n_gpu (int).
        device (torch.device).
    Return:
        predictions (dict).
    """
    if args.architecture == 'TRADE':
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            o, g = model(input_ids, segment_ids, input_masks, 9)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    elif args.architecture == 'SUMBT':
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]
        with torch.no_grad():
            _, pred_slot = model(input_ids, segment_ids, input_masks, labels = None, n_gpu = n_gpu)
        
        for i in range(target_ids.size(0)):
            states = processor.recover_state(pred_slot.tolist()[i], num_turns[i])
        
            for (idx, state) in enumerate(states):
                predictions[guids[i] + f'-{idx}'] = state
    return predictions

def inference(args : argparse.Namespace, model : nn.Module, eval_loader : DataLoader, processor : Any, n_gpu : int, device : torch.device) -> dict:
    """
    Inference.
    Args:
        args (argparse.Namespace).
        model (nn.Module).
        eval_loader (DataLoader).
        processor (Any).
        n_gpu (int).
        device (torch.device).
    Return:
        predictions (dict).
    """
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        predictions = _inference(args, model, batch, processor, predictions, n_gpu, device)
    return predictions

def SomDst_inference(model, eval_examples, processor, slot_meta, tokenizer, op_code = '4'):
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    predictions = {}
    model.eval()
    for i in tqdm(eval_examples):
        turn_id = int(i.guid.split('-')[-1])

        if turn_id == 0:
            last_dialog_state = {}

        i.last_dialog_state = deepcopy(last_dialog_state)
        test_feature = processor._convert_example_to_feature(i)
        test_dataset = WOSDataset([test_feature])
        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, collate_fn=processor.collate_fn)

        for batch in test_loader:
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, state_position_ids, _,\
            _, _, _, _ = batch

        MAX_LENGTH = 20
        with torch.no_grad():
            d, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            )

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []

        pred_ops = [id2op[a] for a in op_ids.tolist()]

        gold_gen = {}
        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code, gold_gen)

        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        predictions[i.guid] = pred_state

    return predictions

def main(args : argparse.Namespace) -> None:
    """
    Inference and Save.
    Args:
        args (argparse.Namespace).
    Return:
        None.
    """
    ontology = json.load(open(f"/opt/ml/input/data/train_dataset/ontology.json"))
    model_dir_path = os.path.dirname(os.path.join(args.model_dir, args.model_name))
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    config = argparse.Namespace(**config)
    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))

    if config.architecture == 'TRADE' or config.architecture == 'SOMDST':
        user_first, dialogue_level = False, False
    elif config.architecture == 'SUMBT':
        user_first, dialogue_level = True, True

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)

    if config.architecture == 'TRADE':
        processor = TRADEPreprocessor(slot_meta, tokenizer)
    elif config.architecture == 'SUMBT':
        processor = SUMBTPreprocessor(slot_meta,
                                    tokenizer,
                                    ontology = ontology,  # predefined ontology
                                    max_seq_length = config.max_seq_length,  # 각 turn마다 최대 길이
                                    max_turn_length = config.max_turn)  # 각 dialogue의 최대 turn 길이
    elif config.architecture == 'SOMDST':
        processor = SOMDSTPreprocessor(
            slot_meta,
            tokenizer,
            max_seq_length=config.max_seq_length
        )


    eval_examples = get_examples_from_dialogues(
        eval_data, user_first = user_first, dialogue_level = dialogue_level, tokenizer = tokenizer
    )

    # Extracting Featrues
    if config.architecture == 'TRADE' or config.architecture == 'SUMBT':
        eval_features = processor.convert_examples_to_features(eval_examples)
        eval_data = WOSDataset(eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_loader = DataLoader(
            eval_data,
            batch_size = args.eval_batch_size,
            sampler = eval_sampler,
            collate_fn = processor.collate_fn,
        )
    print("# eval:", len(eval_examples))

    n_gpu = 1 if torch.cuda.device_count() < 2 else torch.cuda.device_count()

    model = load_model(args, config, tokenizer, ontology, slot_meta, device)
    model.to(device)
    print("Model is loaded")

    if config.architecture == 'TRADE' or config.architecture == 'SUMBT':
        predictions = inference(config, model, eval_loader, processor, n_gpu, device)
    elif config.architecture == 'SOMDST':
        predictions = SomDst_inference(model, eval_examples, processor, slot_meta, tokenizer)
    save_dir = os.path.join(args.output_dir, os.path.dirname(args.model_name))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    json.dump(
        predictions,
        open(f"{save_dir}/predictions.json", "w"),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = '/opt/ml/input/data/eval_dataset')
    parser.add_argument("--model_dir", type = str, default = '/opt/ml/models')
    parser.add_argument("--model_name", type = str, default = 'TRADE/model-2.pth')
    parser.add_argument("--output_dir", type = str, default = '/opt/ml/outputs')
    parser.add_argument("--eval_batch_size", type = int, default = 8)
    args = parser.parse_args()
    
    print(args)

    main(args)