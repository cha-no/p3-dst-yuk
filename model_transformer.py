"""TRADE Transformer decoder version

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, BertModel, BertTokenizer, BertConfig
from transformer.my_transformer import MultiHeadAttention, PositionWiseFeedForward, SinusoidalPositionalEncodedEmbedding

import pdb

def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    '''
    Args:
        logits (Tensor): shape (batch_size, J, max_len, vocab_size).
        target (Tensor): shape (batch_size*J*max_len). 
    '''
    mask = target.ne(pad_idx) 
    logits_flat = logits.view(-1, logits.size(-1)) 
    assert logits_flat.size(0) == target.size(0), f"logtis_flat.size(0) must match with target.size(0) -> {logits_flat.size(0)} != {target.size(0)}"
    log_probs_flat = torch.log(logits_flat) 
    target_flat = target.view(-1, 1) 
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) 
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class TRADE(nn.Module):
    def __init__(self, config, tokenized_slot_meta, pad_idx=0):
        super(TRADE, self).__init__()

        if config.model_name_or_path:
            self.encoder = BertModel.from_pretrained(config.model_name_or_path)
        else:
            self.encoder = BertModel(config)

        self.decoder = TransformerDecoder(config)

        self.decoder.set_slot_idx(tokenized_slot_meta)  
        self.tie_weight()
        

    def tie_weight(self):
        '''tie encoder embedding table weight with decoder embedding tabel weight
        '''

        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(
        self, 
        input_ids,
        target_ids,
        token_type_ids=None, 
        attention_mask=None,    
        ):
        '''
        Args:
            input_ids(Tensor) : shape (batch_size, src_len)
            target_ids(Tensor) : shape (batch_size, trg_len)
            token_type_ids(Tensor) : shape (batch_size, src_len)
            attention_mask(Tensor) : shape (batch_size, src_len)
        Returns:
            all_point_outputs(Tensor) : shape (batch_size, J, max_len, vocab_size)
            all_gate_outputs : shape (batch_size, J, n_gate)
        '''

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids, 
                                                      token_type_ids=token_type_ids, 
                                                      attention_mask=attention_mask)
        ## (batch_size, seq_len, hidden_size), (batch_size, hidden_size)
        all_point_outputs, all_gate_outputs = self.decoder(input_ids,
                                                           target_ids,
                                                           encoder_outputs,
                                                           attention_mask,
                                                           )
        
        return all_point_outputs, all_gate_outputs
    
    def predict(self,
                input_ids,
                max_len,
                token_type_ids=None, 
                attention_mask=None
                ):
        '''
        Args:
            input_ids(Tensor) : shape (batch_size, src_len)
            token_type_ids(Tensor) : shape (batch_size, src_len)
            attention_mask(Tensor) : shape (batch_size, src_len)
            max_len(Int) : maximum number of decoding steps.
        Returns:
            all_point_outputs: shape (batch_size, J, max_len, vocab_size)
            all_gate_outputs: shape (batch_size, J, n_gate)
        '''
        
        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids, 
                                                      token_type_ids=token_type_ids, 
                                                      attention_mask=attention_mask)
        
        all_point_outputs, all_gate_outputs = self.decoder.predict(input_ids,
                                                                   max_len,
                                                                   encoder_outputs,
                                                                   attention_mask,
                                                                   )
        
        return all_point_outputs, all_gate_outputs    



    
class DecoderLayer(nn.Module):
    def __init__(self, 
                 config):
        '''Initialize decoder layer
        
        Args:
            config (Config): configuration parameters.
        '''
        
        super().__init__()
        self.attention_drop_out = config.attention_drop_out
        
        # masked multi_head attention
        self.self_attn = MultiHeadAttention(emb_dim = config.hidden_size,
                                            num_heads = config.num_attention_heads,
                                            drop_out = config.attention_drop_out,
                                            causal = True)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # encoder-decoder attention
        self.enc_dec_attn = MultiHeadAttention(emb_dim = config.hidden_size,
                                               num_heads = config.num_attention_heads,
                                               drop_out = config.attention_drop_out,
                                               encoder_decoder_attention = True)
        self.enc_dec_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        
        #position-wise feed forward
        self.position_wise_feed_forward = PositionWiseFeedForward(config.hidden_size,
                                                               config.ffn_dim,
                                                               config.attention_drop_out)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size)
    
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
        x = F.dropout(x, p = self.attention_drop_out, training = self.training)
        x = self.self_attn_layer_norm(x + residual)
        
        # encoder-decoder attention
        residual = x 
        x, enc_dec_attn_weights = self.enc_dec_attn(query = x,
                                                    key = encoder_output,
                                                    attention_mask = enc_dec_attention_padding_mask)
        
        x = F.dropout(x, p = self.attention_drop_out, training = self.training)
        x = self.enc_dec_attn_layer_norm(x + residual) 
        
        # position-wise feed forward
        residual = x
        x = self.position_wise_feed_forward(x)
        x = self.feed_forward_layer_norm(x + residual)
        
        return x, self_attn_weights, enc_dec_attn_weights



class TransformerDecoder(nn.Module):
    
    def __init__(self, 
                 config, pad_idx=0, proj_dim=None):
        '''Initialize stack of Decoder layers
        
        Args:
            config (Config):Configuration parameters.

        '''
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_idx = pad_idx
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(config.hidden_size, config.proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = config.proj_dim if config.proj_dim else config.hidden_size


        self.n_gate = config.n_gate
        self.dropout = nn.Dropout(config.attention_drop_out)
        self.w_gen = nn.Linear(self.hidden_size, 1) 
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, config.n_gate) 
        
        self.attention_drop_out = config.attention_drop_out
        
        self.embed_positions = SinusoidalPositionalEncodedEmbedding(config.max_position,
                                                                    self.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        
    def set_slot_idx(self, slot_vocab_idx):
        '''padding slot indices vector.
        '''
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole
    
    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x
    
    def generate_causal_mask(self,  
                             size: int,
                             is_inference= False):
        '''Generate padding mask and causal mask
        
        Args:
            size (Int): size of causal mask.
        
        Returns:
            causal_mask (Tensor): shape '(size, size)'
        '''
        

        tmp = torch.ones(size, size, dtype = torch.bool)
        causal_mask = torch.tril(tmp,-1).transpose(0,1).contiguous().to(self.device)
        return causal_mask

    def predict(self,
                input_ids,
                max_len,
                encoder_output=None, 
                input_masks=None
                ):
        '''
        Args:
            input_ids(Tensor) : shape (batch_size, src_len)
            max_len(Int) : maximum number of decoding steps.
            encoder_output (Tensor): output of encoder. shape '(batch_size, src_len, emb_dim)'
            attention_mask(Tensor) : shape (batch_size, src_len)
        Returns:
            all_point_outputs: shape (batch_size, J, max_len, vocab_size)
            all_gate_outputs: shape (batch_size, J, n_gate)
        '''
        
        input_masks = input_masks.ne(1) 

        
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device) 

        slot_e = torch.sum(self.embedding(slot), 1)  # (J, embedding_dim)

        batch_size = encoder_output.size(0)
        J, hidden_size = slot_e.size()
        
        slot_e = slot_e.repeat(batch_size, 1) # (J*batch_size, hidden_size)
        input_masks = input_masks.repeat_interleave(J, dim=0)

        input_ids = input_ids.repeat_interleave(J, dim=0)

        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        
        decoder_input = torch.zeros(J*batch_size, max_len, hidden_size).to(
            input_ids.device
        ) # (J*batch_size, max_len, hidden_size)
        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        )
        
        decoder_input[:,0,:] = slot_e
        for trg_index in range(0, max_len):
            decoder_input_temp = decoder_input[:,:trg_index+1,:] # (J*batch_size, trg_index+1, hidden_size)
            pos_embed = self.embed_positions(torch.zeros(J*batch_size, trg_index)) # (trg_index+1, hidden_size)
            decoder_input_temp += pos_embed
            causal_mask = self.generate_causal_mask(trg_index+1)
            for decoder_layer in self.layers:
                decoder_input_temp, _, _ = decoder_layer(decoder_input_temp, 
                                                                    encoder_output,
                                                                    input_masks,
                                                                    causal_mask)
            '''decoder_input_temp shape (J*batch_size, trg_index, hidden_size)
            '''
            
            
            decoder_output_temp = decoder_input_temp[:,trg_index,:] # (J*batch_size, hidden_size)
            
        
#             attn_e = torch.bmm(encoder_output, decoder_output_temp.unsqueeze(-1))
#             '''(J*batch_size, src_len, hidden_size) x (J*batch_size, hidden_size, 1)
#             -> (J*batch_size, src_len, 1)
#             '''
#             attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)  ## (J*batch_size, src_len)
#             attn_history = F.softmax(attn_e, -1)  ## (J*batch_size, src_len)
            
            
            attn_v = torch.matmul(decoder_output_temp, self.embed.weight.transpose(0, 1))
            '''(J*batch_size, hidden_size) x (hidden_size, vocab_size)
            -> (J*batch_size, vocab_size)
            '''
            attn_vocab = F.softmax(attn_v, -1)

#             p_gen = self.sigmoid(
#                     self.w_gen(decoder_output_temp)
#                 ) # (J*batch_size, 1)
            
            

#             p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
#             ## (J*batch_size, vocab_size)
#             p_context_ptr.scatter_add_(1, input_ids, attn_history)
#             '''
#             p_context_ptr[i][input_ids[i][j]] += attn_history[i][j], 0<=i<=J*batch_size, 
#             0<=j<=seq_len.
#             --> (J*batch_size, vocab_size)
#             '''

#             p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr
            
#             '''attn_vocab shape (J*batch_size, vocab_size)
#             p_context_ptr shape (J*batch_size, vocab_size)
#             p_gen shape = (J*batch_size, 1)
#             '''
            p_final = attn_vocab
            all_point_outputs[:,:,trg_index,:] = p_final.view(batch_size, J, self.vocab_size)
            
            output_indices = torch.argmax(p_final, dim = -1) # (J*batch_size, 1)
            output_indices_embed = self.embed(output_indices)#.unsqueeze(1)
            # shape of self.embed(output_indices) -> (J*batch_size, 1, hidden_size)
            if trg_index < max_len-1:
                decoder_input[:,trg_index+1,:] = output_indices_embed
                
            if trg_index == 0:
                gated_logit = self.w_gate(decoder_output_temp)
                '''decoder_output_temp shape -> (J*batch_size, hidden_size)
                gated_logit shape -> (J*batch_size, n_gate)
                '''
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
                ## (batch_size, J, n_gate)
        
        

        
        return all_point_outputs, all_gate_outputs
    
    
    def forward(self,
                input_ids: torch.Tensor,
                target_ids: torch.Tensor,
                encoder_output: torch.Tensor,
                input_masks: torch.Tensor = None,):
        '''
        Args:
            input_ids (Tensor): dialogue context token ids. shape '(batch_size, src_len)'
            target_ids (Tensor): input to decoder. shape '(batch_size, J = num_slot, trg_len)'
            encoder_output (Tensor): output of encoder. shape '(batch_size, src_len, emb_dim)'
            input_masks (Tensor): Binary BoolTensor for masking padding of encoder output. shape '(batch_size, src_len)'.

        
        Returns:
            x (Tensor): output of decoder. shape '(batch_size, trg_len, emb_dim)'
            enc_dec_attn_weigths (list): list of enc-dec attention weights of each Decoder layer.
        '''

        trg_len = target_ids.size(-1)
        causal_mask = self.generate_causal_mask(trg_len+1) # shape (trg_len+1, trg_len+1)
        

        
        input_masks = input_masks.ne(1) 


        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device) 

        slot_e = torch.sum(self.embedding(slot), 1)  # (J, embedding_dim)

        batch_size = encoder_output.size(0)
        J, hidden_size = slot_e.size()
        
        decoder_input = torch.zeros(J*batch_size, trg_len+1, hidden_size).to(input_ids.device) # (J*batch_size, trg_len+1, hidden_size)
        slot_e = slot_e.repeat(batch_size, 1) # (J*batch_size, hidden_size)
        target_ids = target_ids.reshape(J*batch_size, -1) ## (J*batch_size, trg_len)
        targets_embed = self.embed(target_ids) ## (J*batch_size, trg_len, hidden_size)
        
        decoder_input[:,1:,:] = targets_embed
        decoder_input[:,0,:] = slot_e

        
        pos_embed = self.embed_positions(target_ids) ## (J*batch_size, trg_len+1, hidden_size)
        decoder_input = decoder_input + pos_embed
        decoder_input = F.dropout(decoder_input, p = self.attention_drop_out, training = self.training)
        
        input_ids = input_ids.repeat_interleave(J, dim=0)
        '''(J*batch_size, src_len)
        '''        
        
        input_masks = input_masks.repeat_interleave(J, dim=0)

        encoder_output = encoder_output.repeat_interleave(J, dim=0)

        
        enc_dec_attn_weights = []
        for decoder_layer in self.layers:
            decoder_input, _, attn_weights = decoder_layer(decoder_input, 
                                                           encoder_output,
                                                           input_masks,
                                                           causal_mask)
            '''decoder_input shape (J*batch_size, trg_len+1, hidden_size)
               attn_weights shape (J*batch_size, # attn head, trg_len, src_len)
            '''
        
        decoder_output = decoder_input[:,:-1,:] # (J*batch_size, trg_len, hidden_size)




#         attn_e = torch.bmm(decoder_output, encoder_output.transpose(-1,-2))
#         '''(J*batch_size, trg_len, hidden_size) x (J*batch_size, hidden_size, src_len)
#         -> (J*batch_size, trg_len, src_len)
#         '''

#         attn_e = attn_e.masked_fill(input_masks.unsqueeze(1), -1e9)  ## (J*batch_size, trg_len, src_len)
#         attn_history = F.softmax(attn_e, -1)  ## (J*batch_size, trg_len, src_len)
        
        attn_v = torch.matmul(decoder_output, self.embed.weight.transpose(0, 1))
        '''(J*batch_size, trg_len, hidden_size) x (hidden_size, vocab_size)
        -> (J*batch_size, trg_len, vocab_size)
        '''
        attn_vocab = F.softmax(attn_v, -1)
        
        
#         p_gen = self.sigmoid(
#                 self.w_gen(decoder_output)
#             ) # (J*batch_size, trg_len, 1)
        
#         p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
#         ## (J*batch_size, trg_len, vocab_size)
#         p_context_ptr.scatter_add_(2, input_ids.unsqueeze(1).repeat(1,trg_len,1), attn_history)
#         '''attn_history shape (J*batch_size, trg_len, src_len).
#         input_ids.unsqueeze(1).repeat(1,trg_len,1) shape -> (J*batch_size, trg_len, src_len).

        
#         p_context_ptr[i][j][input_ids[i][j][k]] += attn_history[i][j][k]
#         '''
        
#         p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr
#         '''attn_vocab shape (J*batch_size, trg_len, vocab_size)
#         p_context_ptr shape (J*batch_size, trg_len, vocab_size)
#         p_gen shape = (J*batch_size, trg_len, 1)
#         '''
        p_final = attn_vocab
        all_point_outputs = p_final.view(batch_size, J, trg_len, -1)
        ## (batch_size, J, trg_len, vocab_size)
        
        gated_logit = self.w_gate(decoder_output[:,0,:])
        '''decoder_output[:,0,:] shape -> (J*batch_size, hidden_size)
        gated_logit shape -> (J*batch_size, n_gate)
        '''
        all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
        ## (batch_size, J, n_gate)
        return all_point_outputs, all_gate_outputs