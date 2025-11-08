import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        Q = Q.view(Q.size(0),Q.size(1),self.num_heads,-1).permute(0,2,1,3)
        K = K.view(K.size(0),K.size(1),self.num_heads,-1).permute(0,2,1,3)
        V = V.view(V.size(0),V.size(1),self.num_heads,-1).permute(0,2,1,3)

        attn_scores = Q @ K.transpose(-2,-1) / math.sqrt(self.d_model)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = attn_probs @ V
        attn_output = attn_output.permute(0,2,1,3).contiguous()
        attn_output = attn_output.view(attn_output.size(0),attn_output.size(1),-1)
        return self.W_o(attn_output)

class PostionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PostionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.W_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.W_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):

        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        attn_out = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out = self.cross_attention(x,encoder_output,encoder_output,src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout

        self.pos_encoding = PostionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask = None):
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout

        self.pos_encoding = PostionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers, d_model, num_heads, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout

        # Embedding 层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, max_len, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = generate_mask(src, tgt)
        
        
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc(decoder_output)
        return output


def generate_mask(src, tgt):
    device = src.device 
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2).to(device)
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool().to(device)
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask
