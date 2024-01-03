import torch.nn as nn

from layers.Transformer_layers.embeddings import (DataEmbedding,
                                                  DataEmbeddingNoPosAndTemp,
                                                  DataEmbeddingNoPosition,
                                                  DataEmbeddingNoTemporal)
from layers.Transformer_layers.self_attentions import (AttentionLayer,
                                                       FullAttention)
from layers.Transformer_layers.Transformer_EncDec import (Decoder,
                                                          DecoderLayer,
                                                          Encoder,
                                                          EncoderLayer)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        emb_map = {
            0: DataEmbedding,
            1: DataEmbeddingNoPosition,
            2: DataEmbeddingNoTemporal,
            3: DataEmbeddingNoPosAndTemp
        }

        self.enc_embedding = emb_map[configs.embedding_type](
            configs.enc_input, configs.d_model, configs.embedding, configs.freq, configs.dropout)
        self.dec_embedding = emb_map[configs.embedding_type](
            configs.dec_input, configs.d_model, configs.embedding, configs.freq, configs.dropout)

        # 编码器
        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    attention_layer=AttentionLayer(
                        attention=FullAttention(mask_flag=False, factor=configs.factor,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention),
                        d_model=configs.d_model, n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model, d_ff=configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ], 
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 解码器
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        attention=FullAttention(mask_flag=True, factor=configs.factor,
                                                attention_dropout=configs.dropout,
                                                output_attention=False),
                        d_model=configs.d_model, n_heads=configs.n_heads
                    ),
                    cross_attention=AttentionLayer(
                        attention=FullAttention(mask_flag=False, factor=configs.factor,
                                                attention_dropout=configs.dropout,
                                                output_attention=False),
                        d_model=configs.d_model, n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model, d_ff=configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.d_layers)
            ], 
            norm_layer=nn.LayerNorm(configs.d_model), 
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]