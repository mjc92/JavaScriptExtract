''' Define the Layers '''
import torch.nn as nn
from models.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import time

__author__ = "Yu-Hsiang Huang"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # initialize multi head attention
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        start = time.time()
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        # remove positional feedforward network as it consumes too much time
        # enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn # [mb x len_v x d_model], [mb*hea