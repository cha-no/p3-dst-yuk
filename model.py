"""DST Models

    TRADE, SUMBT, SOMDST

"""

import argparse
from typing import List, Dict, Tuple

import os.path

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers.modeling_bert import BertOnlyMLMHead

def masked_cross_entropy_for_value(logits : torch.Tensor, target : torch.Tensor, pad_idx : int = 0) -> torch.Tensor:
    """Generation loss

    Args:
        logits (torch.Tensor)
        target (torch.Tensor)
        pad_idx (int, optional). Defaults to 0.

    Returns:
        torch.Tensor: loss
    """
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss

class TRADE(nn.Module):
    """TRADE Model"""

    def __init__(self : nn.Module, config : argparse.Namespace, tokenized_slot_meta : List[int], pad_idx : int = 0) -> None:
        """initialize TRADE

        Args:
            self (nn.Module)
            config (argparse.Namespace)
            tokenized_slot_meta (List[int])
            pad_idx (int, optional). Defaults to 0.
        """
        super(TRADE, self).__init__()
        self.config = config
        self.tokenized_slot_meta = tokenized_slot_meta

        # encoder
        if config.model_name_or_path:
            self.encoder = BertModel.from_pretrained(config.model_name_or_path)
        else:
            self.encoder = BertModel(config)

        # decoder
        self.decoder = SlotGenerator(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.n_gate,
            config.proj_dim,
            pad_idx,
        )
        # set slot index tokenized slot meta
        self.decoder.set_slot_idx(self.tokenized_slot_meta)
        
        # pretrain masked language model
        self.mlm_head = BertOnlyMLMHead(self.config)
        
        # set equal decoder embedding weight and encoder embedding weight
        self.tie_weight()
        
    def tie_weight(self : nn.Module) -> None:
        """equal decoder embedding weight and encoder embedding weight

        Args:
            self (nn.Module)
        """
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(
        self : nn.Module,
        input_ids : torch.Tensor,
        token_type_ids : torch.Tensor,
        attention_mask : torch.Tensor = None,
        max_len : int = 10,
        teacher : torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TRADE forward

        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            token_type_ids (torch.Tensor): segment id
            attention_mask (torch.Tensor, optional)
            max_len (int, optional): max length of generation tokens. Defaults to 10.
            teacher (torch.Tensor, optional): target_ids. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: generation distribution, gate distribution
        """
        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids, 
                                                      token_type_ids=token_type_ids, 
                                                      attention_mask=attention_mask)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids,
            encoder_outputs,
            pooled_output.unsqueeze(0),
            attention_mask,
            max_len,
            teacher,
        )

        return all_point_outputs, all_gate_outputs
    
    @staticmethod
    def mask_tokens(
        inputs : torch.Tensor, 
        tokenizer : BertTokenizer, 
        config : argparse.Namespace, 
        mlm_probability : float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        Args:
            inputs (torch.Tensor)
            tokenizer (BertTokenizer)
            config (argparse.Namespace)
            mlm_probability (float, optional). Defaults to 0.15.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input_id, labels
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
        #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]

        probability_matrix.masked_fill_(torch.eq(labels, 0), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(device=inputs.device, dtype=torch.bool) & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(device=inputs.device, dtype=torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(config.vocab_size, labels.shape, device=inputs.device, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random].to(inputs.device)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def forward_pretrain(
        self : nn.Module, 
        input_ids : torch.Tensor,
        tokenizer : BertTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pretrain forward

        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            tokenizer (BertTokenizer)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logits, labels
        """
        input_ids, labels = self.mask_tokens(input_ids, tokenizer, self.config)
        encoder_outputs, _ = self.encoder(input_ids=input_ids)
        mlm_logits = self.mlm_head(encoder_outputs)
        
        return mlm_logits, labels


class GRUEncoder(nn.Module):
    """TRADE Encoder

    Args:
        nn.Module
    """
    def __init__(self, vocab_size, d_model, n_layer, dropout, proj_dim=None, pad_idx=0):
        super(GRUEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if proj_dim:
            self.proj_layer = nn.Linear(d_model, proj_dim, bias=False)
        else:
            self.proj_layer = None

        self.d_model = proj_dim if proj_dim else d_model
        self.gru = nn.GRU(
            self.d_model,
            self.d_model,
            n_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        mask = input_ids.eq(self.pad_idx).unsqueeze(-1)
        x = self.embed(input_ids)
        if self.proj_layer:
            x = self.proj_layer(x)
        x = self.dropout(x)
        o, h = self.gru(x)
        o = o.masked_fill(mask, 0.0)
        output = o[:, :, : self.d_model] + o[:, :, self.d_model :]
        hidden = h[0] + h[1]  # n_layer 고려
        return output, hidden


class SlotGenerator(nn.Module):
    """TRADE Decoder

    Args:
        nn.Module
    """

    def __init__(
        self : nn.Module, 
        vocab_size : int, 
        hidden_size : int, 
        dropout : float, 
        n_gate : int, 
        proj_dim : int = None, 
        pad_idx : int = 0
    ):
        """Initialize TRADE Decoder

        Args:
            self (nn.Module)
            vocab_size (int)
            hidden_size (int): embedding hidden size
            dropout (float)
            n_gate (int): number of gate. ex) dontcare, none, yes, no, ptr
            proj_dim (int, optional): projection dimension. Defaults to None.
            pad_idx (int, optional): padding index. Defaults to 0.
        """
        super(SlotGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = proj_dim if proj_dim else hidden_size

        # generation
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate)

    def set_slot_idx(self : nn.Module, slot_vocab_idx : List[int]) -> None:
        """set slot index

        Args:
            self (nn.Module)
            slot_vocab_idx (List[int]): slot tokenizing index
        """
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self : nn.Module, x : torch.Tensor) -> torch.Tensor:
        """embedding

        Args:
            self (nn.Module): [description]
            x (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
        self :nn.Module, 
        input_ids : torch.Tensor, 
        encoder_output : torch.Tensor, 
        hidden : torch.Tensor, 
        input_masks : torch.Tensor, 
        max_len : int, 
        teacher : torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TRADE Decoder forward

        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            encoder_output (torch.Tensor): encoder(bert) output
            hidden (torch.Tensor): encoder(bert) output
            input_masks (torch.Tensor)
            max_len (int): max length of generation tokens.
            teacher (torch.Tensor, optional): target_ids. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (generation distribution, gate distribution)
        """
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)
        slot_e = torch.sum(self.embedding(slot), 1)  # J,d
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size, device = input_ids.device)
        
        # Parallel Decoding
        w = slot_e.repeat(batch_size, 1).unsqueeze(1)
        hidden = hidden.repeat_interleave(J, dim=1)
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        input_ids = input_ids.repeat_interleave(J, dim=0)
        input_masks = input_masks.repeat_interleave(J, dim=0)
        for k in range(max_len):
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D

            # B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
            attn_history = F.softmax(attn_e, -1)  # B,T

            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            )  # B,V
            attn_vocab = F.softmax(attn_v, -1)

            # B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  # B,1
            p_gen = p_gen.squeeze(-1)

            p_context_ptr = torch.zeros_like(attn_vocab, device = input_ids.device)
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            _, w_idx = p_final.max(-1)

            if teacher is not None:
                # w = self.embedding(teacher[:, :, k]).transpose(0, 1).reshape(batch_size * J, 1, -1)
                w = self.embedding(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
            else:
                w = self.embedding(w_idx).unsqueeze(1)  # B,1,D
            if k == 0:
                gated_logit = self.w_gate(context.squeeze(1))  # B,3
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)

        return all_point_outputs, all_gate_outputs


class BertForUtteranceEncoding(BertPreTrainedModel):
    """SUMBT Encoder

    Args:
        BertPreTrainedModel
    """
    def __init__(self : BertPreTrainedModel, config : argparse.Namespace) -> None:
        """Initialize SUMBT Encoder

        Args:
            self (BertPreTrainedModel)
            config (argparse.Namespace)
        """
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(
        self : BertPreTrainedModel, 
        input_ids : torch.Tensor, 
        token_type_ids : torch.Tensor, 
        attention_mask : torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder forward

        Args:
            self (BertPreTrainedModel)
            input_ids (torch.Tensor)
            token_type_ids (torch.Tensor)
            attention_mask (torch.Tensor)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: bert output
        """
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )


class MultiHeadAttention(nn.Module):
    """Multi head attention

    Args:
        nn.Module
    """
    def __init__(self : nn.Module, heads : int, d_model : int, dropout : float = 0.1) -> None:
        """Initialize multi head attention

        Args:
            self (nn.Module)
            heads (int): number of multi head
            d_model (int): dimension of model
            dropout (float, optional). Defaults to 0.1.
        """
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(
        self : nn.Module, 
        q : torch.Tensor, 
        k : torch.Tensor, 
        v : torch.Tensor, 
        d_k : int, 
        mask : torch.Tensor = None, 
        dropout : float = None
    ) -> torch.Tensor:
        """Attention Mechanism

        Args:
            self (nn.Module)
            q (torch.Tensor): query
            k (torch.Tensor): key
            v (torch.Tensor): value
            d_k (int): scale number
            mask (torch.Tensor, optional): attention mask. Defaults to None.
            dropout (float, optional). Defaults to None.

        Returns:
            torch.Tensor: attention value
        """

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(
        self : nn.Module, 
        q : torch.Tensor, 
        k : torch.Tensor, 
        v : torch.Tensor, 
        mask : torch.Tensor = None
    ) -> torch.Tensor:
        """multi head attention forward

        Args:
            self (nn.Module)
            q (torch.Tensor): query
            k (torch.Tensor): key
            v (torch.Tensor): value
            mask (torch.Tensor, optional): attention mask. Defaults to None.

        Returns:
            torch.Tensor: attention output
        """
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class SUMBT(nn.Module):
    """SUMBT Model

    Args:
        nn.Module
    """
    def __init__(self : nn.Module, args : argparse.Namespace, num_labels : List[int], device : torch.device) -> None:
        """Initialize SUMBT

        Args:
            self (nn.Module)
            args (argparse.Namespace)
            num_labels (List[int]): number of slot's labels
            device (torch.device)
        """
        super(SUMBT, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList(
            [nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels]
        )

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)

        ### RNN Belief Tracker
        self.nbt = nn.GRU(
            input_size=self.bert_output_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_num_layers,
            dropout=self.hidden_dropout_prob,
            batch_first=True,
        )
        self.init_parameter(self.nbt)

        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob),
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self : nn.Module, label_ids : List[torch.Tensor], slot_ids : torch.Tensor) -> None:
        """make slot value lookup vector

        Args:
            self (nn.Module)
            label_ids (List[torch.Tensor]) list token ids of slot value
            slot_ids (torch.Tensor) token ids of slots
        """

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(
            slot_ids.device
        )
        slot_mask = slot_ids > 0
        hid_slot, _ = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        )
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(
                label_id.device
            )
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(
                label_id.view(-1, self.max_label_length),
                label_type_ids.view(-1, self.max_label_length),
                label_mask.view(-1, self.max_label_length),
            )
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")
        self.sv_encoder = None

    def forward(
        self : nn.Module,
        input_ids : torch.Tensor,
        token_type_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        labels : torch.Tensor = None,
        n_gpu : int = 1,
        target_slot : List[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SUMBT forward

        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            token_type_ids (torch.Tensor)
            attention_mask (torch.Tensor)
            labels (torch.Tensor, optional). Defaults to None.
            n_gpu (int, optional). Defaults to 1.
            target_slot (List[int], optional): target index. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (loss, loss_slot, acc, acc_slot, pred_slot)
        """
        # input_ids: [B, M, N]
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # labels: [B, M, J]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden, _ = self.utterance_encoder(
            input_ids.view(-1, self.max_seq_length),
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        )
        hidden = torch.mul(
            hidden,
            attention_mask.view(-1, self.max_seq_length, 1)
            .expand(hidden.size())
            .float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*M*B, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [J*M*B, N, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        )
        hidden = hidden.squeeze()  # h [J*M*B, H] Aggregated Slot Context
        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(
                self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
            ).to(
                self.device
            )  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h)

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [J*B, M, H_GRU]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(
                self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
            ).to(
                self.device
            )  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = rnn_out.view(slot_dim, ds, ts, -1)  # [J, B, M, H]

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = (
                hid_label.unsqueeze(0)
                .unsqueeze(0)
                .repeat(ds, ts, 1, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _hidden = (
                hidden[s, :, :, :]
                .unsqueeze(2)
                .repeat(1, 1, num_slot_labels, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
            _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        pred_slot = torch.cat(pred_slot, 2)
        if labels is None:
            return output, pred_slot

        # calculate joint accuracy
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = (
            torch.sum(accuracy, 0).float()
            / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        )
        acc = (
            sum(torch.sum(accuracy, 1) / slot_dim).float()
            / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()
        )  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return (
                loss.unsqueeze(0),
                None,
                acc.unsqueeze(0),
                acc_slot.unsqueeze(0),
                pred_slot.unsqueeze(0),
            )

    @staticmethod
    def init_parameter(module : nn.Module) -> None:
        """Initialize GRU

        Args:
            module (nn.Module)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)
            

class SomDST(nn.Module):
    """SOMDST

    Args:
        nn.Module
    """
    def __init__(
        self : nn.Module, 
        config : argparse.Namespace,
        n_op : int, 
        n_domain : int, 
        update_id : int, 
        exclude_domain : bool = False
    ) -> None:
        """Initialize SOMDST

        Args:
            self (nn.Module)
            config (argparse.Namespace)
            n_op (int): operation number. ex) dontcare, delete, carryover, update
            n_domain (int): number of domains
            update_id (int): update id. ex) 3
            exclude_domain (bool, optional). Defaults to False.
        """
        super(SomDST, self).__init__()
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(config, n_op, n_domain, update_id, exclude_domain)
        self.decoder = Decoder(config, self.encoder.bert.embeddings.word_embeddings.weight)

    def forward(
        self : nn.Module, 
        input_ids : torch.Tensor, 
        token_type_ids : torch.Tensor,
        state_positions : torch.Tensor, 
        attention_mask : torch.Tensor,
        max_value : int, 
        op_ids : int = None, 
        max_update : int = None, 
        teacher : torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SOMDST forward

        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            token_type_ids (torch.Tensor)
            state_positions (torch.Tensor)
            attention_mask (torch.Tensor)
            max_value (int): max length of slot's tokens
            op_ids (int, optional): operation ids. Defaults to None.
            max_update (int, optional): max update operation in batches. Defaults to None.
            teacher (torch.Tensor, optional): target_ids. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (domain scores, state scores, generation scores)
        """
        
        # input_ids : (batch_size, max_seq_length)
        # token_type_ids : (batch_size, max_seq_length)
        # state_positions : (batch_size, num_slots)
        # attention_mask : (batch_size, max_seq_length)
        # op_ids : (batch_size, num_slots)
        # max_update : batch단위로 update된 slot의 최대 갯수
        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask,
                                   op_ids=op_ids,
                                   max_update=max_update)

        # domain_scores : (batch_size, n_domain)
        # state_scores : (batch_size, num_slots, n_op)
        # decoder_input : (batch_size, max_update, hidden_size)
        # sequence_output : (batch_size, max_seq_length, hidden_size)
        # pooled_output : (1, batch_size, hidden_size)
        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs
    
        # max_value : batch단위로 slot을 token화 했을 때 최대 갯수
        gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
                                  pooled_output, max_value, teacher)

        # gen_scores : (batch_size, max_update, max_value, vocab_size)
        return domain_scores, state_scores, gen_scores


class Encoder(nn.Module):
    """SOMDST Encoder

    Args:
        nn.Module
    """
    def __init__(
        self : nn.Module, 
        config : argparse.Namespace, 
        n_op : int, 
        n_domain : int, 
        update_id : int, 
        exclude_domain : bool = False
    ) -> None:
        """Initialize SOMDST Encoder

        Args:
            self (nn.Module)
            config (argparse.Namespace)
            n_op (int): number of operations ex). this case is 4 (delete, carryover, dontcare, update) 
            n_domain (int): number of domains
            update_id (int): this case is 3('update' : 3)
            exclude_domain (bool, optional). Defaults to False.
        """
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel.from_pretrained(config.model_name_or_path)
        self.bert.resize_token_embeddings(config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, n_op)   # operation action을 분류하는 layer
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)   # domain을 분류하는 layer
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id   # operation dictionary에서 update에 해당하는 id 여기서는 1임

    def forward(
        self : nn.Module, 
        input_ids : torch.Tensor, 
        token_type_ids : torch.Tensor,
        state_positions : torch.Tensor, 
        attention_mask : torch.Tensor,
        op_ids : int = None, 
        max_update : int = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SOMDST Encoder forward

        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            token_type_ids (torch.Tensor)
            state_positions (torch.Tensor)
            attention_mask (torch.Tensor)
            op_ids (int, optional): operation ids. Defaults to None.
            max_update (int, optional): max update operation in batches. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (domain score, state score, decoder input, bert output, hidden)
        """
        # bert_outputs : (sequence_output, pooled_output)
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)

        # sequence_outputs : (batch_size, max_seq_len, hidden_size)
        # pooled_outputs : (batch_size, hidden_size)
        sequence_output, pooled_output = bert_outputs[:2]

        # state_pos : (batch_size, num_slots, hidden_size) state_position을 hidden_size만큼 확장        
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))

        # state_output : (batch_size, num_slots, hidden_size)
        state_output = torch.gather(sequence_output, 1, state_pos)

        # state_score : (batch_size, num_slots, n_op) operation을 분류하는 분포     
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4
        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            # domain_scores : (batch_size, n_domain)
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        batch_size = state_scores.size(0)
        if op_ids is None:
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()

        gathered = []
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            # b : (num_slots, hidden_size) 각 slot에 해당하는 벡터
            # a : (num_slots) slot에서 update된 slot을 true로 한 값
            if a.sum().item() != 0:
                # v : (update된 slot의 갯수(데이터마다 다름), hidden_size)
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                # max_update로 크기를 맞춰주기위해 zero padding을 함
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            # v : (1, max_update, hidden_size)
            gathered.append(v)

        # decoder_input : (batch_size, max_update, hidden_size)
        decoder_inputs = torch.cat(gathered)
        return domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output.unsqueeze(0)


class Decoder(nn.Module):
    """SOMDST Decoder

    Args:
        nn.Module
    """
    def __init__(self : nn.Module, config : argparse.Namespace, bert_model_embedding_weights : torch.Tensor) -> None:
        """Initialize SOMDST Decoder

        Args:
            self (nn.Module)
            config (argparse.Namespace)
            bert_model_embedding_weights (torch.Tensor): Encoder embedding weight
        """
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

        # set weight
        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                p.data.normal_(mean=0.0, std=config.initializer_range)


    def forward(
        self : nn.Module, 
        x : torch.Tensor, 
        decoder_input : torch.Tensor, 
        encoder_output : torch.Tensor, 
        hidden : torch.Tensor, 
        max_len : int, 
        teacher : torch.Tensor = None
    ) -> torch.Tensor:
        """SOMDST Decoder forward

        Args:
            self (nn.Module)
            x (torch.Tensor): input_ids
            decoder_input (torch.Tensor)
            encoder_output (torch.Tensor): encoder bertoutput
            hidden (torch.Tensor): encoder bertoutput hidden
            max_len (int): max length of slot's tokens
            teacher (torch.Tensor, optional): target ids. Defaults to None.

        Returns:
            torch.Tensor: [description]
        """
        # mask : (batch_size, max_seq_len)
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long

        # state_in : (batch_size, max_update, hidden_size)
        state_in = decoder_input

        # all_point_outputs : (max_update, batch_size, max_value, vocab_size) 이부분 TRADE처럼 분포를 만들고 값을 채워나가는 것 같음
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):
            # w : (batch_size, 1, hidden_size)
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)

                # hidden : (1, batch_size, hidden_size)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # _, (hidden, _) = self.lstm(w, (hidden, hidden))

                # attn_e : (batch_size, max_seq_len) hidden과 encoder_output을 attention
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)

                # attn_history : (batch_size, max_seq_len) attn_e를 softmax해서 history 분포를 구함
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # attn_v : (batch_size, vocab_size) hidden과 embedding weight를 attention
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V

                # attn_vocab : (batch_size, vocab_size) attn_v를 softmax해서 vocab 분포를 구함
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # context : (batch_size, 1, hidden_size)
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                # p_gen : (batch_size, 1) w와 hidden과 context를 concat하고 layer를 거쳐 분포 혼합비율을 구합니다
                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                # p_context_ptr : (batch_size, vocab_size) scatter_add를 사용해 size를 맞춰줌
                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V

                # p_final : (batch_size, vocab_size) history분포와 vocab분포를 혼합
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                
                all_point_outputs[j, :, k, :] = p_final   # 구한 분포를 채워나갑니다

        return all_point_outputs.transpose(0, 1)