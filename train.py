import argparse
from importlib import import_module
from pathlib import Path

import os
import sys
sys.path.append(os.path.abspath('./KoBERT-Transformers'))

import glob
import json

import re
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import AdamW, BertTokenizer, BertConfig, get_linear_schedule_with_warmup
from tokenization_kobert import KoBertTokenizer

import wandb

from data_utils import (
    WOSDataset,
    get_examples_from_dialogues,
    load_dataset,
    convert_state_dict, 
    DSTInputExample, 
    OpenVocabDSTFeature, 
    DSTPreprocessor,
    SOMDSTFeature,
    EXPERIMENT_DOMAINS,
    OP_SET,
    domain2id,
    set_seed
)

from eval_utils import DSTEvaluator, AverageMeter
from evaluation import _evaluation
from inference import inference, SomDst_inference
from model import *
from preprocessor import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
op_code = '4'
op2id = OP_SET[op_code]

def set_wandb(args : argparse.Namespace) -> None:
    """
    Set Wandb.
    Args:
        args (argparse.Namespace).
    Return:
        None.
    """
    config = {}
    for arg in dir(args):
        if not arg.startswith('_') and arg not in ['data_dir', 'model_dir']:
            config[arg] = getattr(args, arg)

    wandb.init(
        project = 'exp', 
        config = config,
    )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def increment_path(path : str, exist_ok : bool = False) -> str:
    """
    Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    Return:
        Path to save model (str)
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def set_model(args : argparse.Namespace, tokenizer : BertTokenizer, ontology : dict, slot_meta : list, device : torch.device) -> nn.Module:
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
    if args.architecture == 'TRADE':
        # Slot Meta tokenizing for the decoder initial inputs
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens = False)
            )
        
        # Model 선언
        config = BertConfig.from_pretrained(args.model_name_or_path)
        config.model_name_or_path = args.model_name_or_path
        config.n_gate = args.n_gate
        config.proj_dim = args.proj_dim
        model = TRADE(config, tokenized_slot_meta)
        # model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화
        # print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
        print("Model is initialized")
    elif args.architecture == 'SUMBT':
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, args.max_label_length)
        num_labels = [len(s) for s in slot_values_ids]
        args.num_labels = num_labels
        model = SUMBT(args, num_labels, device)
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV
    elif args.architecture == 'SOMDST':
        config = BertConfig.from_pretrained(args.model_name_or_path)
        config.model_name_or_path = args.model_name_or_path
        config.dropout = args.dropout
        config.attention_probs_dropout_prob = args.dropout
        config.hidden_dropout_prob = args.dropout
        config.vocab_size = len(tokenizer)
        args.vocab_size = len(tokenizer)
        args.initializer_range = config.initializer_range
        model = SomDST(config, len(op2id), len(domain2id), op2id['update'])
        model.encoder.bert.embeddings.word_embeddings.weight.data[35000].normal_(mean=0.0, std=0.02)
        model.encoder.bert.embeddings.word_embeddings.weight.data[35001].normal_(mean=0.0, std=0.02)
        model.encoder.bert.embeddings.word_embeddings.weight.data[35002].normal_(mean=0.0, std=0.02)
    return model

def get_loss(args : argparse.Namespace, model : nn.Module, batch : int, n_gpu : int, tokenizer : BertTokenizer, 
            device : torch.device, loss_fnc_1, loss_fnc_2) -> torch.Tensor:
    """
    Calculate loss based on model training.
    Args:
        args (argparse.Namespace).
        model (nn.Module).
        batch (int).
        n_gpu (int).
        tokenizer (BertTokenizer): "dsksd/bert-ko-small-minimal".
        device (torch.device).
        loss_fnc_1: use TRADE
        loss_fnc_2: use TRADE
    Return:
        Calculate loss (torch.Tensor).
    """
    if args.architecture == 'TRADE':
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]
        # teacher forcing
        if (
            args.teacher_forcing_ratio > 0.0
            and random.random() < args.teacher_forcing_ratio
        ):
            tf = target_ids
        else:
            tf = None

        all_point_outputs, all_gate_outputs = model(
            input_ids, segment_ids, input_masks, target_ids.size(-1), tf
        )
        
        # generation loss
        loss_1 = loss_fnc_1(
            all_point_outputs.contiguous(),
            target_ids.contiguous().view(-1),
            tokenizer.pad_token_id,
        )
        # gating loss
        loss_2 = loss_fnc_2(
            all_gate_outputs.contiguous().view(-1, args.n_gate),
            gating_ids.contiguous().view(-1),
        )
        loss = loss_1 + loss_2
        return loss, loss_1, loss_2

    elif args.architecture == 'SUMBT':
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids  = \
        [b.to(device) if not isinstance(b, list) else b for b in batch]

        # Forward
        if n_gpu == 1:
            loss, loss_slot, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
        else:
            loss, _, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
    
        return loss

    if args.architecture == 'SOMDST':
        batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
        input_ids, input_mask, segment_ids, state_position_ids, op_ids,\
        domain_ids, gen_ids, max_value, max_update = batch
        # teacher forcing
        if (
            args.teacher_forcing_ratio > 0.0
            and random.random() < args.teacher_forcing_ratio
        ):
            tf = gen_ids
        else:
            tf = None

        domain_scores, state_scores, gen_scores = model(input_ids=input_ids,
                                                        token_type_ids=segment_ids,
                                                        state_positions=state_position_ids,
                                                        attention_mask=input_mask,
                                                        max_value=max_value,
                                                        op_ids=op_ids,
                                                        max_update=max_update,
                                                        teacher=tf)
        loss_s = loss_fnc_2(state_scores.view(-1, len(op2id)), op_ids.view(-1))
        loss_g = loss_fnc_1(gen_scores.contiguous(),
                            gen_ids.contiguous(),
                            tokenizer.vocab['[PAD]'])
        loss = loss_s + loss_g

        if args.exclude_domain is not True:
            loss_d = loss_fnc_2(domain_scores.view(-1, len(domain2id)), domain_ids.view(-1))
            loss = loss + loss_d
        else:
            loss_d = 0

        return loss, loss_s, loss_g, loss_d


def get_lr(optimizer : transformers.optimization) -> float:
    """
    Get learning_rate.
    Args:
        optimizer (transformers.optimization).
    Return:
        Learning_Rate (float).
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def delete_model(model_dir : str, score : float, score_dict : dict, number : int) -> bool:
    """
    Delete model to save best number models
    Args:
        model_dir (str).
        score (float).
        score_dict (dict).
        number (int). : num of models
    Return:
        flag (bool). : decide to save model
    """
    if len(score_dict) >= number :
        min_file_name, min_score = min(score_dict.items(), key = lambda x : x[1])
        if score > min_score:
            score_dict.pop(min_file_name)
            os.remove(os.path.join(model_dir, min_file_name))
            return True
        else:
            return False
    else:
        return True

# def delete_model(model_dir : str) -> None:
#     """
#     Delete Model to save best Model
#     Args:
#         model_dir (str).
#     Return:
#         None.
#     """
#     file_list = os.listdir(model_dir)
#     for file in file_list:
#         if file.startswith('model'):
#             os.remove(os.path.join(model_dir, file))
#             break

def mlm_pretrain(args, model, loader, optimizer, tokenizer, loss_fnc_pretrain, epoch):
    model.train()
    for step, batch in enumerate(loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [b.to(device) if not isinstance(b, list) else b for b in batch]

        logits, labels = model.forward_pretrain(input_ids, tokenizer)
        loss = loss_fnc_pretrain(logits.view(-1, args.vocab_size), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            current_lr = get_lr(optimizer)
            if args.mlm_during:
                epochs = args.num_train_epochs
            elif args.mlm_during:
                epochs = args.num_pretrain_epochs
            print(
                f"[{epoch + 1}/{epochs}] [{step}/{len(loader)}] loss : {loss.item()}  lr : {current_lr}"
            )


def train(args : argparse.Namespace) -> None:
    """
    Training Model.
    Args:
        args (argparse.Namespace).
    Return:
        None.
    """
    # random seed 고정
    set_seed(args.random_seed)

    # wandb 연동
    set_wandb(args)

    # model_dir설정
    model_dir = increment_path(os.path.join(args.model_dir, args.architecture))

    wandb.run.name = model_dir

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    ontology = json.load(open(f"{args.data_dir}/ontology.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    # Define tokenizer
    model_name_or_path = args.model_name_or_path
    if model_name_or_path == "monologg/kobert":
        tokenizer = KoBertTokenizer.from_pretrained(model_name_or_path)
    else:
        #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    if args.architecture == 'TRADE' or args.architecture == 'SOMDST':
        user_first, dialogue_level = False, False
    elif args.architecture == 'SUMBT':
        user_first, dialogue_level = True, True

    train_examples = get_examples_from_dialogues(
        train_data, slot_meta, user_first = user_first, dialogue_level = dialogue_level, tokenizer = tokenizer
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, slot_meta, dev_labels, user_first = user_first, dialogue_level = dialogue_level, tokenizer = tokenizer
    )

    # Define processor
    if args.architecture == 'TRADE':
        processor = TRADEPreprocessor(slot_meta, 
                                    tokenizer,
                                    max_seq_length = args.max_seq_length,
                                    word_dropout = args.word_dropout)
        args.vocab_size = len(tokenizer)
        args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr
    elif args.architecture == 'SUMBT':
        max_turn = max([len(e['dialogue']) for e in train_data])
        processor = SUMBTPreprocessor(slot_meta,
                                    tokenizer,
                                    ontology = ontology,  # predefined ontology
                                    max_seq_length = args.max_seq_length,  # 각 turn마다 최대 길이
                                    max_turn_length = max_turn,
                                    word_dropout = args.word_dropout)  # 각 dialogue의 최대 turn 길이
        args.max_turn = max_turn
    elif args.architecture == 'SOMDST':
        processor = SOMDSTPreprocessor(
            slot_meta,
            tokenizer,
            max_seq_length=args.max_seq_length
        )


    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    model = set_model(args, tokenizer, ontology, slot_meta, device)
    model.to(device)

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size = args.train_batch_size,
        sampler = train_sampler,
        collate_fn = processor.collate_fn,
        num_workers = args.num_workers
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size = args.eval_batch_size,
        sampler = dev_sampler,
        collate_fn = processor.collate_fn,
    )
    print("# dev:", len(dev_data))

    n_gpu = 1 if torch.cuda.device_count() < 2 else torch.cuda.device_count()
    n_epochs = args.num_train_epochs
    t_total = len(train_loader) * n_epochs

    # Optimizer 및 Scheduler 선언
    if args.architecture == 'SOMDST':
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        enc_param_optimizer = list(model.encoder.named_parameters())
        enc_optimizer_grouped_parameters = [
            {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        warmup_steps = int(t_total * args.warmup_ratio)

        enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.learning_rate)
        enc_scheduler = get_linear_schedule_with_warmup(
                            enc_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
                        )

        dec_param_optimizer = list(model.decoder.parameters())
        dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_learning_rate)
        dec_scheduler = get_linear_schedule_with_warmup(
                            dec_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
                        )

    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params" : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay" : args.weight_decay,
                },
                {
                    "params" : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay" : 0.0,
                },
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)

        warmup_steps = int(t_total * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = warmup_steps, num_training_steps = t_total
        )

    # TRADE, SOMDST 일 때 사용
    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating
    
    # MLM pretrain
    loss_fnc_pretrain = nn.CrossEntropyLoss()  # MLM pretrain

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    json.dump(
        vars(args),
        open(f"{model_dir}/exp_config.json", "w"),
        indent = 2,
        ensure_ascii = False,
    )
    json.dump(
        slot_meta,
        open(f"{model_dir}/slot_meta.json", "w"),
        indent = 2,
        ensure_ascii = False,
    )
    
    # Train
    print("This notebook use [%s]."%(device))
    
    if args.mlm_pre:
        print('mlm_pretrain!!!')
        for epoch in range(args.num_pretrain_epochs):
            mlm_pretrain(args, model, train_loader, optimizer, tokenizer, loss_fnc_pretrain, epoch)

    score_dict = {}
    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = AverageMeter()

        for step, batch in enumerate(train_loader):
            if args.architecture == 'TRADE':
                loss, loss_1, loss_2 = get_loss(args, model, batch, n_gpu, tokenizer, device, loss_fnc_1, loss_fnc_2)
            elif args.architecture == 'SUMBT':
                loss = get_loss(args, model, batch, n_gpu, tokenizer, device, loss_fnc_1, loss_fnc_2)
            elif args.architecture == 'SOMDST':
                loss, loss_s, loss_g, loss_d = get_loss(args, model, batch, n_gpu, tokenizer, device, loss_fnc_1, loss_fnc_2)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.architecture == 'SOMDST':
                enc_optimizer.step()
                enc_scheduler.step()
                dec_optimizer.step()
                dec_scheduler.step()
                model.zero_grad()
                wandb.log({"learning_rate" : get_lr(enc_optimizer)})
                wandb.log({"dec_learning_rate" : get_lr(dec_optimizer)})
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                wandb.log({"learning_rate" : get_lr(optimizer)})

            train_loss.update(loss.item(), len(batch))

            if step % 100 == 0:
                if args.architecture == 'TRADE':
                    current_lr = get_lr(optimizer)
                    print(
                        f"[{epoch + 1}/{n_epochs}] [{step}/{len(train_loader)}] loss : {loss.item()} gen : {loss_1.item()} gate : {loss_2.item()}  lr : {current_lr}"
                    )
                    wandb.log({"Train loss" : train_loss.avg})
                    wandb.log({"Train gen loss" : loss_1.item()})
                    wandb.log({"Train gate loss" : loss_2.item()})
                elif args.architecture == 'SUMBT':
                    current_lr = get_lr(optimizer)
                    print(
                        f"[{epoch + 1}/{n_epochs}] [{step}/{len(train_loader)}] loss : {loss.item()}  lr : {current_lr}"
                    )
                    wandb.log({"Train loss" : train_loss.avg})
                elif args.architecture == 'SOMDST':
                    enc_current_lr = get_lr(enc_optimizer)
                    dec_current_lr = get_lr(dec_optimizer)
                    if args.exclude_domain is not True:
                        print(
                            f"[{epoch + 1}/{n_epochs}] [{step}/{len(train_loader)}] loss : {loss.item()} state_loss : {loss_s.item()} gen_loss : {loss_g.item()} dom_loss : {loss_d.item()} enc_lr : {enc_current_lr} dec_lr : {dec_current_lr}"
                        )
                        wandb.log({"Train loss" : train_loss.avg})
                        wandb.log({"Train gen loss" : loss_g.item()})
                        wandb.log({"Train state loss" : loss_s.item()})
                        wandb.log({"Train domain loss" : loss_d.item()})
                    else:
                        print(
                            f"[{epoch + 1}/{n_epochs}] [{step}/{len(train_loader)}] loss : {loss.item()} state_loss : {loss_s.item()} gen_loss : {loss_g.item()} enc_lr : {enc_current_lr} dec_lr : {dec_current_lr}"
                        )
                        wandb.log({"Train loss" : train_loss.avg})
                        wandb.log({"Train gen loss" : loss_g.item()})
                        wandb.log({"Train state loss" : loss_s.item()})

        if args.mlm_during:
            print('mlm_pretrain!!!')
            mlm_pretrain(args, model, train_loader, optimizer, tokenizer, loss_fnc_pretrain, epoch)

        if args.architecture == 'SOMDST':
            predictions = SomDst_inference(model, dev_examples, processor, slot_meta, tokenizer)
        else:
            predictions = inference(args, model, dev_loader, processor, n_gpu, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")
            wandb.log({f"{k}" : v, 'epochs' : epoch + 1}, commit = False)

        score = eval_result['joint_goal_accuracy']
        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch + 1
        
        if delete_model(model_dir, score, score_dict, 5):    
            score_dict[f'model-{epoch + 1}.bin'] = eval_result['joint_goal_accuracy']
            torch.save(model.state_dict(), f"{model_dir}/model-{epoch + 1}.bin")
        
    print(f"Best checkpoint: {model_dir}/model-{best_checkpoint}.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--random_seed", type = int, default = 2020)
    parser.add_argument("--data_dir", type = str, default = "/opt/ml/input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default = "models")
    parser.add_argument("--architecture", type = str, default = "TRADE")
    parser.add_argument("--num_workers", type = int, default = 1)
    parser.add_argument("--group_decay", type = bool, default = True)
    parser.add_argument("--weight_decay", type = float, default = 0.01)
    parser.add_argument("--max_seq_length", type = int, default = 512)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--word_dropout", type = float, default = 0.0)
    parser.add_argument("--max_label_length", type = int, default = 12)
    parser.add_argument("--train_batch_size", type = int, default = 8)
    parser.add_argument("--eval_batch_size", type = int, default = 8)
    parser.add_argument("--mlm_pre", type = bool, default = False)
    parser.add_argument("--mlm_during", type = bool, default = False)
    parser.add_argument("--learning_rate", type = float, default = 5e-5)
    parser.add_argument("--adam_epsilon", type = float, default = 1e-8)
    parser.add_argument("--max_grad_norm", type = float, default = 1.0)
    parser.add_argument("--num_train_epochs", type = int, default = 10)
    parser.add_argument("--num_pretrain_epochs", type = int, default = 3)
    parser.add_argument("--warmup_ratio", type = int, default = 0.1)
    parser.add_argument("--teacher_forcing_ratio", type = float, default = 0.5)
    parser.add_argument(
        "--model_name_or_path",
        type = str,
        help = "Subword Vocab만을 위한 huggingface model",
        default = "monologg/koelectra-base-v3-discriminator",
    )

    # TRADE Architecture Specific Argument
    parser.add_argument("--hidden_size", type = int, help = "GRU의 hidden size", default = 768)
    # parser.add_argument(
    #     "--vocab_size",
    #     type = int,
    #     help = "vocab size, subword vocab tokenizer에 의해 특정된다",
    #     default = None,
    # )
    parser.add_argument("--hidden_dropout_prob", type = float, default = 0.1)
    parser.add_argument("--proj_dim", type = int,
                        help = "만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.", default=None)

    # SUMBT Architecture Specific Argument
    parser.add_argument("--hidden_dim", type = int, help = "GRU의 hidden dimension", default = 300)
    parser.add_argument("--num_rnn_layers", type = int, help = "GRU의 rnn layers", default = 1)
    parser.add_argument("--zero_init_rnn", type = bool, help = "이부분 아직 모르겠는데 baseline과 같게 했습니다", default = False)
    parser.add_argument("--attn_head", type = int, help = "SUMBT Attention head 갯수", default = 4)
    parser.add_argument("--fix_utterance_encoder", type = bool, help = "utterance_encoder의 weight를 freeze 결정", default = False)
    parser.add_argument("--task_name", type = str, help = "이부분 baseline에서 쓰지 않았는데 일단 있어서 추가했습니다", default = 'sumbtgru')
    parser.add_argument("--distance_metric", type = str, help = "freeze된 value vector와의 거리를 구할 때 metric", default = 'euclidean')

    # SOMDST Architecture Specific Argument
    parser.add_argument("--dec_learning_rate", type = float, default = 1e-4)
    parser.add_argument("--exclude_domain", type = str2bool, default = True)

    args = parser.parse_args()

    print(args)

    train(args)