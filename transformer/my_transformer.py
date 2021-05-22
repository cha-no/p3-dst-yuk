"""Transformer

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

import time
import math

import matplotlib.pyplot as plt

import tqdm




class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 num_heads: int, 
                 drop_out: float = 0.0,
                 bias: bool = False, 
                 encoder_decoder_attention: bool = False,
                 causal: bool = False):
        '''Initialize MultiHeadAttention class variables.
        
        Args:
            emb_dim (int): Dimension of a word * number of heads.
            num_heads (int): Number of attention heads.
            drop_out (float): Drop out rate.
            bias (bool): Boolean that indicating whether to use bias or not.
            encoder_decoder_attention (bool): Boolean that indicating whether the multi head
                                              attention is encoder-decoder attention or not.
            causal (bool): Boolean that indicating whether to use causal mask or not.
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        
        self.drop_out = drop_out
        self.encoder_decoder_attention = encoder_decoder_attention
        self.causal = causal
        
        self.wk = nn.Linear(self.emb_dim, self.emb_dim, bias = bias)
        self.wq = nn.Linear(self.emb_dim, self.emb_dim, bias = bias)
        self.wv = nn.Linear(self.emb_dim, self.emb_dim, bias = bias)
        self.output = nn.Linear(self.emb_dim, self.emb_dim, bias = bias)
    
    def multi_head_scaled_dot_product(self,
                                      query: torch.Tensor,
                                      key: torch.Tensor,
                                      value: torch.Tensor,
                                      attention_mask: torch.BoolTensor):
        '''Perform multi-head version of scaled dot product.
        
        Args:
            query (Tensor): shape '(batch size, # attention head, seqence length, demension of head)'
            key (Tensor): shape '(batch size, # attention head, seqence length, demension of head)'
            value (Tensor): shape '(batch size, # attention head, seqence length, demension of head)'
            attention_mask: This mask can be either causal mask or padding mask.
                            shape '(batch size, source squence length)' for padding mask.
                            shape '(sequence length, target sequence length)' for causal mask.
        Returns:
            attn_output (Tensor): output of attention mechanism. shape '(batch size, seq_len, emb_dim)'
            attn_weights (Tensor): value of attention weight of each word. shape '(batch size, # attn head, seq_len, seq_len)'
        '''
        
        attn_weights = torch.matmul(query, key.transpose(-1,-2)) / math.sqrt(self.head_dim)
        '''shape of attn_weights : (batch size, # attn head, seq_len, seq_len)'''
        
        if attention_mask is not None:
            if self.causal:
                '''Masking future info for encoder-decoder attention.'''
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
                '''
                shape of attention_mask : (trg_len, trg_len).
                shape of attention_mask.unsqueeze(0).unsqueeze(1) : (1, 1, trg_len, trg_len).
                '''
            else:
                '''Masking padding token so that it is not used for attention.'''
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                '''
                shape of attention_mask : (batch_size, src_len)
                shape of attention_mask.unsqueeze(1).unsqueeze(2) : (batch_size, 1, 1, src_len)
                '''
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_probs = F.dropout(attn_weights, p=self.drop_out, training=self.training)
        
        attn_output = torch.matmul(attn_probs, value)
        '''shape of attn_output : (batch size, # attn head, seq_len, head_dim)'''
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        '''shape of attn_output : (batch size, seq_len, # attn head, head_dim)'''
        shape = attn_output.size()[:-2] + (self.emb_dim,)
        attn_output = attn_output.view(*shape)
        '''shape of attn_output : (batch size, seq_len, emb_dim)'''
        attn_output = self.output(attn_output)
        
        return attn_output, attn_weights
    
    def transform_to_multi_head(self, 
                                x: torch.Tensor):
        ''' Reshape input
        
        Args:
            x (Tensor): shape '(batch_size, seq_len, emb_dim)'
        
        Returns:
            Tensor: shape '(batch_size, # attn head, seq_len, head_dim)'
        '''
        
        shape = x.size()[:-1] + (self.num_heads, self.head_dim,)
        x = x.view(*shape)
        
        return x.permute(0, 2, 1, 3)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                attention_mask: torch.Tensor = None):
        '''
        Args:
            query (Tensor): shape '(batch_size, seq_len, emb_dim)'
            key (Tensor): shape '(batch_size, seq_len, emb_dim)'
            attention_mask (Tensor): shape '(batch size, squence length)' for padding mask.
                                     shape '(sequence length, sequence length)' for causal mask.
        
        Returns:
            attn_output (Tensor): output of attention mechanism. shape '(batch size, seq_len, emb_dim)'
            attn_weights (Tensor): value of attention weight of each word. shape '(batch size, # attn head, seq_l
        '''
        
        q = self.wq(query)
        
        # encoder-decoder attention
        if self.encoder_decoder_attention:
            '''
            query is output of encoder
            key is input of decoder
            '''
            k = self.wk(key)
            v = self.wv(key)
        
        # self attention
        else:
            '''
            Both of query and key are input of encoder(query is same with key).
            '''
            k = self.wk(query)
            v = self.wv(query)
        
        q = self.transform_to_multi_head(q)
        k = self.transform_to_multi_head(k)
        v = self.transform_to_multi_head(v)
        
        attn_output, attn_weights = self.multi_head_scaled_dot_product(q,k,v,attention_mask)
            
        return attn_output, attn_weights

class PositionWiseFeedForward(nn.Module):
    
    def __init__(self,
                 emb_dim: int,
                 hid_dim: int,
                 drop_out: float = 0.1):
        '''Initialize position-wise feed forward network.
        
        Args:
            emb_dim (int): word embdding dimension.
            hid_dim (int): hidden dimesion.
            drop_out (float): drop out rate.
        '''
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, hid_dim)
        self.linear_2 = nn.Linear(hid_dim, emb_dim)
        self.activation = nn.ReLU()
        self.drop_out = drop_out
    
    def forward(self,
                 x: torch.Tensor):
        '''
        Args:
            x (Tensor): shape '(batch_size, seq_len, emb_dim)'
        
        Return:
            x (Tensor): shape '(batch_size, seq_len, emb_dim)'
        '''

        x = self.linear_1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        
        x = self.linear_2(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        
        return x

class SinusoidalPositionalEncodedEmbedding(nn.Embedding):
    def __init__(self, 
                 max_position: int, 
                 embedding_dim: int):
        '''Initialize positional embedding.
        
        Args:
            max_position (int): maximum length of input sequence length.
                                 That is, it can encode position up to max_positions'th position.
            embedding_dim (int): embedding dimension.
        '''
        super().__init__(max_position, embedding_dim)
        self.weight = self._init_weight(self.weight)

    def _init_weight(self, initial_embedding_table: nn.Parameter):
        '''Make positional embedding table
        
        Args:
            initial_embedding_table (Parameter): initialized embedding table.
        
        Returns:
            pe (Parameter): position embedding table.
        
        '''
        max_pos, emb_dim = initial_embedding_table.shape
        pe = nn.Parameter(torch.zeros(max_pos, emb_dim))

        pos_id = torch.arange(0, max_pos).unsqueeze(1)
        freq = torch.pow(10000., -torch.arange(0, emb_dim, 2, dtype=torch.float) / emb_dim)
        pos_freq = pos_id * freq
        pe[:, 0::2] = torch.sin(pos_freq)
        pe[:, 1::2] = torch.cos(pos_freq)
        
        pe.detach_() ## pe가 leaf노드가 되도록 한다.
        # pe.requires_grad = True
        
        return pe

    def forward(self, 
                input_ids: torch.Tensor):
        '''
        Args:
            input_ids (Tensor): shape '(batch_size, seq_len)'
        
        Return:
            Tensor : shape '(seq_len, emb_dim)'
        '''
        batch_size, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len+1, dtype=torch.long, device=self.weight.device)
        ## slot의 자리를 추가해야하므로 seq_len애 +1을 한다.
        return super().forward(positions)

class EncoderLayer(nn.Module):
    def __init__(self, 
                 config):
        '''Initialize encoder layer
        
        Args:
            config (Config): configuration parameters.
        '''
        
        super().__init__()

        self.drop_out = config.drop_out
        
        # self multi-head attention
        self.self_attn = MultiHeadAttention(emb_dim = config.emb_dim,
                                            num_heads = config.num_attention_heads,
                                            drop_out = config.attention_drop_out)                      
        self.attn_layer_norm = nn.LayerNorm(config.emb_dim)
        
        #position-wise feed forward
        self.position_wise_feed_forward = PositionWiseFeedForward(config.emb_dim,
                                                               config.ffn_dim,
                                                               config.drop_out)
        self.feed_forward_layer_norm = nn.LayerNorm(config.emb_dim)
    
    def forward(self, 
                x: torch.Tensor, 
                encoder_padding_mask: torch.Tensor):
        '''
        Args:
            x (Tensor): shape '(batch_size, src_len, emb_dim)'
            encoder_padding_mask (Tensor): binary BoolTensor. shape '(batch_size, src_len)'
            
        Returns:
            x (Tensor): encoded output. shape '(batch_size, src_len, emb_dim)'
            self_attn_weights: self attention socre
        '''
        residual = x
        x, self_attn_weights = self.self_attn(query=x, 
                                              key=x, 
                                              attention_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.drop_out, training = self.training)
        x = self.attn_layer_norm(x + residual)
        
        residual = x
        x = self.position_wise_feed_forward(x)
        x = self.attn_layer_norm(x + residual)
        
#         clamping
        if x.isnan().any() or x.isinf().any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min = -clamp_value, max = clamp_value)
        return x, self_attn_weights

class Encoder(nn.Module):
    def __init__(self, 
                 config, 
                 embedding_table: nn.Embedding):
        '''Initialize stack of Encoder layers
        
        Args:
            config (Config):Configuration parameters.
            embedding_table (nn.Embedding): instance of nn.Embedding for Encoder input tokens.
                                            input tokens shape '(batch_size, src_len)'
                                            embedding table shape '(num_voca, emb_dim)'
        '''
        super().__init__()
        
        self.drop_out = config.drop_out
        
        self.embedding_table = embedding_table
        self.embed_positions = SinusoidalPositionalEncodedEmbedding(config.max_position,
                                                                    config.emb_dim)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])                     
        
    def forward(self, 
                input_indices: torch.Tensor, 
                padding_mask = None):
        '''
        Args:
            input_indices (Tensor): input to Encoder. shape '(batch_size, src_len)'
            padding_mask (Tensor): padding mask. shape '(batch_size, src_len)'
            
        Returns:
            x (Tensor): Encoder output. shape '(batch_size, src_len, emb_dim)'
            self_attn_scores (list): list of attention weights of each Encoder layer.
        '''
        
        inputs_embed = self.embedding_table(input_indices)
        pos_embed = self.embed_positions(input_indices)
        x = inputs_embed + pos_embed
        x = F.dropout(x, p = self.drop_out, training = self.training)
        
        self_attn_weights = []
        for encoder_layer in self.layers:
            x, attn_weights = encoder_layer(x, padding_mask)
            self_attn_weights.append(attn_weights.detach().clone())
        return x, self_attn_weights

class DecoderLayer(nn.Module):
    def __init__(self, 
                 config):
        '''Initialize decoder layer
        
        Args:
            config (Config): configuration parameters.
        '''
        
        super().__init__()
        self.drop_out = config.drop_out
        
        # masked multi_head attention
        self.self_attn = MultiHeadAttention(emb_dim = config.emb_dim,
                                            num_heads = config.num_attention_heads,
                                            drop_out = config.attention_drop_out,
                                            causal = True)
        self.self_attn_layer_norm = nn.LayerNorm(config.emb_dim)
        
        # encoder-decoder attention
        self.enc_dec_attn = MultiHeadAttention(emb_dim = config.emb_dim,
                                               num_heads = config.num_attention_heads,
                                               drop_out = config.attention_drop_out,
                                               encoder_decoder_attention = True)
        self.enc_dec_attn_layer_norm = nn.LayerNorm(config.emb_dim)
        
        #position-wise feed forward
        self.position_wise_feed_forward = PositionWiseFeedForward(config.emb_dim,
                                                               config.ffn_dim,
                                                               config.drop_out)
        self.feed_forward_layer_norm = nn.LayerNorm(config.emb_dim)
    
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                enc_dec_attention_padding_mask: torch.Tensor = None,
                causal_mask: torch.Tensor = None):
        
        '''
        Args:
            x (Tensor): Input to decoder layer. shape '(batch_size, trg_len, emb_dim)'.
            encoder_output (Tensor): Output of encoder. shape '(batch_size, src_len, emb_dim)'
            enc_dec_attention_padding_mask (Tensor): Binary BoolTensor for masking padding of
                                                     encoder output.
                                                     shape '(batch_size, src_len)'.
            causal_mask (Tensor): Binary BoolTensor for masking future information in decoder.
                                  shape '(batch_size, trg_len)'
        
        Returns:
            x (Tensor): Output of decoder layer. shape '(batch_size, trg_len, emb_dim)'.
            self_attn_weights (Tensor): Masked self attention weights of decoder. 
                                        shape '(batch_size, trg_len, trg_len)'.
            enc_dec_attn_weights (Tensor): Encoder-decoder attention weights.
                                           shape '(batch_size, trg_len, src_len)'.
        '''
        
        # msked self attention
        residual = x
        x, self_attn_weights = self.self_attn(query = x,
                                              key = x,
                                              attention_mask = causal_mask)
        x = F.dropout(x, p = self.drop_out, training = self.training)
        x = self.self_attn_layer_norm(x + residual)
        
        # encoder-decoder attention
        residual = x
        x, enc_dec_attn_weights = self.enc_dec_attn(query = x,
                                                    key = encoder_output,
                                                    attention_mask = enc_dec_attention_padding_mask)
        x = F.dropout(x, p = self.drop_out, training = self.training)
        x = self.enc_dec_attn_layer_norm(x + residual)
        
        # position-wise feed forward
        residual = x
        x = self.position_wise_feed_forward(x)
        x = self.feed_forward_layer_norm(x + residual)
        
        return x, self_attn_weights, enc_dec_attn_weights

class TransformerDecoder(nn.Module):
    
    def __init__(self, 
                 config,
                 embedding_table: nn.Embedding):
        '''Initialize stack of Encoder layers
        
        Args:
            config (Config):Configuration parameters.
            embedding_table (nn.Embedding): instance of nn.Embedding for Decoder input tokens.
                                            input tokens shape '(batch_size, trg_len)'
                                            embedding table shape '(num_voca, emb_dim)'
        '''
        
        super().__init__()
        
        self.drop_out = config.drop_out
        
        self.embedding_table = embedding_table
        self.embed_positions = SinusoidalPositionalEncodedEmbedding(config.max_position,
                                                                    config.emb_dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
    
    def generate_causal_mask(self,  
                             trg: torch.LongTensor):
        '''Generate padding mask and causal mask
        
        Args:
            trg (LongTensor): input to decoder. shape '(batch_size, trg_len)'
        
        Returns:
            causal_mask (Tensor): shape '(trg_len, trg_len)'
        '''
        tmp = torch.ones(trg.size(1), trg.size(1), dtype = torch.bool)
        causal_mask = torch.tril(tmp,-1).transpose(0,1).contiguous().to(self.device)
        
        return causal_mask
    
    def generate_padding_mask(self, 
                              src: torch.LongTensor):
        '''Generate padding mask
        
        Args:
            src (LongTensor): input to encoder. shape '(batch_size, src_len)'
        
        Returns:
            padding_mask (Tensor): shape '(batch_size, src_len)'
        '''
        padding_mask = src.eq(self.SRC.vocab.stoi['<pad>']).to(self.device)
        
        return padding_mask
    
    def predict(self,
                src: torch.LongTensor):
        '''
        Args:
            src (LongTensor): input to encoder. shape '(batch_size, src_len)'
        
        Returns:
            output_tokens (LongTensor): predicted tokens. shape'(batch_size, max_position)'
        '''
        padding_mask = self.generate_padding_mask(src)
        
        encoder_output, _ = self.encoder(input_indices = src,
                                         padding_mask = padding_mask)
        output_tokens = (torch.ones((self.config.batch_size, self.config.max_position))\
                         * self.TRG.vocab.stoi['<pad>']).long().to(self.device) 
        ## (batch_size, max_position)
        output_tokens[:,0] = self.TRG.vocab.stoi['<sos>']
        for trg_index in range(1, self.config.max_position):
            trg = output_tokens[:,:trg_index] # (batch_size, trg_index)
            causal_mask = self.generate_causal_mask(trg) # (trg_index, trg_index)
            output, _ = self.decoder(input_indices = trg,
                                     encoder_output = encoder_output,
                                     enc_dec_attention_padding_mask = padding_mask,
                                     causal_mask = causal_mask) # (batch_size, trg_index, emb_dim)
            output = self.linear(output) # (batch_size, trg_index, # trg vocab)
            output = torch.argmax(output, dim = -1) # (batch_size, trg_index)
            output_tokens[:,trg_index] = output[:,-1]
        
        return output_tokens
    
    def forward(self,
                input_indices: torch.Tensor,
                encoder_output: torch.Tensor,
                enc_dec_attention_padding_mask: torch.Tensor = None,
                causal_mask: torch.Tensor = None):
        '''
        Args:
            input_indeces (Tensor): input to decoder. shape '(batch_size, trg_len)'
            encoder_output (Tensor): output of encoder. shape '(batch_size, src_len, emb_dim)'
            enc_dec_attention_padding_masl (Tensor): Binary BoolTensor for masking padding of
                                                     encoder output.
                                                     shape '(batch_size, src_len)'.
            causal_mask (Tensor): Binary BoolTensor for masking future information in decoder.
                                  shape '(batch_size, trg_len)'
        
        Returns:
            x (Tensor): output of decoder. shape '(batch_size, trg_len, emb_dim)'
            enc_dec_attn_weigths (list): list of enc-dec attention weights of each Decoder layer.
        '''
        padding_mask = self.generate_padding_mask(src)
        causal_mask = self.generate_causal_mask(trg)
        
        
        inputs_embed = self.embedding_table(input_indices)
        pos_embed = self.embed_positions(input_indices)
        x = inputs_embed + pos_embed
        x = F.dropout(x, p = self.drop_out, training = self.training)
        
        enc_dec_attn_weights = []
        for decoder_layer in self.layers:
            x, _, attn_weights = decoder_layer(x, 
                                               encoder_output,
                                               enc_dec_attention_padding_mask,
                                               causal_mask)
            enc_dec_attn_weights.append(attn_weights.detach().clone())
        return x, enc_dec_attn_weights

    
    
class SlotGenerator(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
    ):
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

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1) ## p^gen계산에 쓰이는 W_1임.
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate) ## G_j 계산에 쓰이는 W_g임.

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
        self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        '''
        Args:
            input_ids (Tensor) : shape (batch_size, seq_len)
            encoder_output (Tensor) : shape (batch_size, seq_len, hidden_size)
            hidden (Tensor) : shape (1, batch_size, hidden_size)
            input_masks (Tensor) : shape (batch_size, seq_len)
            max_len (Int) : train시에는 target_ids.size(-1)로 주고 inference는 특정 값으로 고정하여 준다.
            teacher (Tensor) : train시에는 특정 확률로 target_id(batch_size, num_slot, target_id 토큰갯수) 이거나 None임. inference시에는 None.
        '''
        input_masks = input_masks.ne(1) ## input_masks의 True와 False를 반전시킨다.(True는 False로, False는 True로)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device) 
        '''domain-slot을 토크나이징하여 얻은 인덱스들에 패딩까지 넣은것이 slot임
        shape (J, 4). 현재 데이터에서는 4가 토크나이징된 domain-slot들 중에 가장 긴 것이다.
        '''
        slot_e = torch.sum(self.embedding(slot), 1)  # (J, embedding_dim = 768)
        '''self.embedding(slot).size() = torch.Size([J, batch_size, hidden_size])
        한 domain-slot에 대한 모든 토큰들의 임베딩벡터를 합친다.
        --> slot_e[0]은 0번째 slot의 embedding vector인 것임.
        '''
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        ) 
        ''' torch.Size([4, 45, 5, 35000]). 각 decoding step에 대해서 35000개의 vocab에 대한 distribution이 나오게 됨. (batch_size, J, max_decoding_step, vocab_size)'''
        # Parallel Decoding -> 모든 slot에 대한 디코딩을 동시에 진행한다.
        w = slot_e.repeat(batch_size, 1).unsqueeze(1) ## 디코더의 첫번째 스텝에서의 input으로 쓰임.
        '''shape (J*batch_size, 1, hidden_size). (J, hidden_size)가 똑같이 batch_size개수 만큼 있게 됨.
        '''
        hidden = hidden.repeat_interleave(J, dim=1) ## 디코더의 initial hidden state로 쓰인다.
        '''shape (1, J*batch_size, hidden_size)
        1차원 방향으로 첫번째 data에 대한 hidden_size크기의 벡터가 똑같은게 J개 있고 그다음
        두번째 data에 대한 hidden_size크기의 벡터가 똑같은게 J개 있고 ...
        그다음 batch_size번째 data에 대한 hidden_size크기의 벡터가 똑같은게 J개 있음.
        '''
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        '''(J*batch_size, seq_len, hidden_size)첫번째 데이터에 대한 (1, seq_len, hidden_size)가 J번 반복되고 
        그 다음 두번째 데이터에 대한 (1, seq_len, hidden_size)가 J번 반복되고 ...
        '''
        input_ids = input_ids.repeat_interleave(J, dim=0)
        '''(J*batch_size, seq_len)
        첫번째 데이터에 대한 input_id가 J번 반복되고 두번째 데이터에 대한 input_id가 J번 반복되고 ...
        '''
        input_masks = input_masks.repeat_interleave(J, dim=0)
        '''(J*batch_size, seq_len)
        첫번째 데이터에 대한 input_mask가 J번 반복되고 두번째 데이터에 대한 input_mask가 J번 반복되고 ...
        
        '''
        for k in range(max_len): ## max_len번 만큼 디코딩한다. max_len = train시에는 target_ids.size(-1).
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D (1,J*batch_size, hidden_size)

            ### p^history 계산. B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            '''(J*batch_size, seq_len, hidden_size)와 (J*batch_size, hidden_size, 1)을 배치 매트릭스곱을 한다. -> (J*batch_size, seq_len, 1)
            '''
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
            '''항상 softmax에 넣기전에 패딩토큰 부분을 마스킹해야 한다. (J*batch_size, seq_len)
            '''
            attn_history = F.softmax(attn_e, -1)  # B,T
            
            
            ### p^vocab 계산.
            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            ) 
            '''(J*batch_size, hidden_size)와 (hidden_size, vocab_size)를 매트릭스곱을 함. --> (J*batch_size, vocab_size)
            '''
            attn_vocab = F.softmax(attn_v, -1)

            # P^gen 계산.  B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # c_jk 계산
            ''' (J*batch_size, 1, seq_len)와 (J*batch_size, seq_len, hidden_size)를 행렬곱
                --> (J*batch_size, 1, hidden_size)
            '''
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  
            ''' (J*batch_size, 1, 1)'''
            p_gen = p_gen.squeeze(-1) 
            ''' (J*batch_size, 1) '''

            p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device) 
            '''(J*batch_size, vocab_size)'''
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            '''
            p_context_ptr[i][input_ids[i][j]] += attn_history[i][j], 0<=i<=J*batch_size, 
            0<=j<=seq_len.
            --> (J*batch_size, vocab_size)
            '''
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            '''(J*batch_size, vocab_size). 각 데이터의 각 slot에 대한 k번째 디코딩 스텝의 확률분포.'''
            _, w_idx = p_final.max(-1) ## shapne [J*batch_size,]

            if teacher is not None:
                '''teacher -> (batch_size, num_slot, target_id 토큰갯수)'''
                w = self.embedding(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
                # rand_idx = torch.randint(5, 34000, w_idx.size()).to(input_ids.device) 
                # w = self.embedding(rand_idx).unsqueeze(1)
                '''self.embedding(teacher[:, :, k]) = 모든 배치데이터의 모든 slot에 대한 value의 k번째 토큰의 embedidng vector(batch_size, num_slot, emb_dim).
                '''
            else:
                w = self.embedding(w_idx).unsqueeze(1)  
                '''(J*batch_size, 1 ,hidden_size)'''
            if k == 0: 
                '''첫번째 스텝에서만 slot_gate값을 계산한다.'''
                gated_logit = self.w_gate(context.squeeze(1))  # (J*batch_size, 3)
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate) ## 모든 slot에 대한 gate값 예측.
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size) ## 모든 slot에 대한 각 decoding step의 vocab 확률분포 예측.

        return all_point_outputs, all_gate_outputs