import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder
from transformer_encoder import TransformerEncoderLayer
import math
from torch.nn.modules.normalization import LayerNorm


# our model

class Trans(nn.Module):
    def __init__(self, d_model, DNN_dim, head, layer, k_dim, v_dim, fed_dim, encoder_type):
        super(Trans, self).__init__()
        self.d_model = d_model
        self.head = head
        self.layer = layer
        self.fed_dim = fed_dim
        self.encoder_type = encoder_type
        self.norm = LayerNorm(d_model)

        if self.encoder_type == "Transformer":
            self.transformer_encoder_layer = TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.head, dim_feedforward=self.fed_dim, kdim=k_dim, vdim=v_dim)
            self.transformer_encoder = TransformerEncoder(
                self.transformer_encoder_layer, num_layers=self.layer, norm=self.norm)

        elif self.encoder_type == "DNN":
            self.encoder = nn.Sequential(
                LayerNorm(self.d_model),
                nn.Linear(self.d_model, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, DNN_dim),
                nn.Sigmoid()
            )

        else:
            print("there maybe mistakes")

        # self.input = torch.arange(0, self.d_model).view(1, -1).long()

    def forward(self, x):
        B, src_len, src_dim = x.shape[0], x.shape[1], x.shape[2]
        # input = torch.repeat_interleave(self.input, x.shape[0], dim=0)
        # enc_outputs = self.src_emb(input.cuda())  # [batch_size, src_len, d_model]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # pad_mask = self.get_attn_pad_mask(x.squeeze(1), x.squeeze(1))
        # trans_emb = self.transformer_encoder(enc_outputs.to(device) * x.to(device).transpose(1, 2), mask=pad_mask.cuda()).sum(1)
        if self.encoder_type == "Transformer":
            trans_emb = self.transformer_encoder(x).view(B, -1)
        elif self.encoder_type == "DNN":
            trans_emb = self.encoder(x.squeeze(1))
        else:
            trans_emb = None
            print("there maybe mistakes")
        return trans_emb

    def get_attn_pad_mask(self, seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

# office model
class Ori_Trans(nn.Module):
    def __init__(self, d_model, head, layer, fed_dim):
        super(Ori_Trans, self).__init__()
        self.d_model = d_model
        self.head = head
        self.layer = layer
        self.fed_dim = fed_dim

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.head, dim_feedforward=self.fed_dim)

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.layer)
        self.transformer_encoder = torch.nn.DataParallel(self.transformer_encoder)

    def forward(self, x, device):
        return self.transformer_encoder(x.to(device)).squeeze(1)


if __name__ == '__main__':
    input = torch.rand(128, 1, 4096).cuda()
    d_model = input.shape[-1]
    model = Trans(d_model, head=2, layer=2, k_dim=128, v_dim=128, fed_dim=512).cuda()
    n_p = sum(x.numel() for x in model.parameters())
    print("model net par", n_p)
    print("model net:", model)
    trans_emb = model(input)
    print(trans_emb.shape)



