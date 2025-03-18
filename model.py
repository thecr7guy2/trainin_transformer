import torch
import math


class InputEmbeddings(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddingss = torch.nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddingss(x) * math.sqrt(self.d_model)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(self.seq_len, self.d_model)

        for i in range(self.seq_len):
            for j in range(self.d_model):
                denom = torch.pow(torch.tensor(10000.0), (2 * j) / self.d_model)
                num = torch.tensor(float(i))
                if j % 2 == 0:
                    pe[i, j] = torch.sin(num / denom)
                else:
                    pe[i, j] = torch.cos(num / denom)

        pe = pe.unsqueeze(0)
        print(pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormm(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(features, eps=1e-5)

    def forward(self, x):
        return self.layer_norm(x)


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, dff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(dff, d_model)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MHA(torch.nn.Module):
    def __init__(self, d_model, number_of_heads, dropout):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.d_model = d_model
        self.noh = number_of_heads

        self.dk = self.d_model // self.noh

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.wo = torch.nn.Linear(d_model, d_model)

    @staticmethod
    def calculate_self_attention(qprime, kprime, vprime, mask, dropout):
        dk = qprime.shape[-1]
        attention_scores = (qprime @ kprime.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        # why last dim ?
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ vprime), attention_scores

    def forward(self, q, k, v, mask):
        qprime = self.wq(q)
        # (batch,seq_length,dmodel)
        kprime = self.wk(k)
        # (batch,seq_length,dmodel)
        vprime = self.wv(v)
        # (batch,seq_length,dmodel)

        qprime = qprime.view(qprime.shape[0], qprime.shape[1], self.noh, self.dk)
        # (batch,seq_length,dmodel) =>(batch,seq_length,noh,dk)
        qprime = qprime.transpose(1, 2)
        # (batch,seq_length,noh,dk) => (batch,noh,seq_length,dk)

        kprime = kprime.view(kprime.shape[0], kprime.shape[1], self.noh, self.dk)
        kprime = kprime.transpose(1, 2)

        vprime = vprime.view(vprime.shape[0], vprime.shape[1], self.noh, self.dk)
        vprime = vprime.transpose(1, 2)

        x, attention_scores = MHA.calculate_self_attention(
            qprime, kprime, vprime, mask, self.dropout
        )
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.noh * self.dk)
        return self.wo(x)


class SkipConnection(torch.nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm = LayerNormm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))


class EncoderBlock(torch.nn.Module):
    def __init__(self, features, mha_block, feedforward_block, dropout):
        super().__init__()
        self.attention_block = mha_block
        self.feedforward_block = feedforward_block
        self.skip_connections = torch.nn.ModuleList(
            [SkipConnection(features, dropout) for _ in range(2)]
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.skip_connections[0](
            x, lambda x: self.attention_block(x, x, x, src_mask)
        )
        x = self.skip_connections[1](x, self.feedforward_block)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, features: int, layers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, features, mha_block, mha_block2, feedforward_block, dropout):
        super().__init__()
        self.attention_block = mha_block
        self.cross_attention_block = mha_block2
        self.feedforward_block = feedforward_block
        self.skip_connections = torch.nn.ModuleList(
            [SkipConnection(features, dropout) for _ in range(3)]
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.skip_connections[0](
            x, lambda x: self.attention_block(x, x, x, tgt_mask)
        )
        x = self.skip_connections[1](
            x, lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask)
        )
        x = self.skip_connections[2](x, self.feedforward_block)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, features: int, layers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormm(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(torch.nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


class Transformer(torch.nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        src_pos_enc,
        tgt_pos_enc,
        src_emb,
        tgt_emb,
        projection_layer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos_enc = src_pos_enc
        self.tgt_pos_enc = tgt_pos_enc
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos_enc(src)
        x = self.encoder(src, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos_enc(tgt)
        x = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return x

    def project(self, x):
        x = self.projection_layer(x)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Standard forward: encode the src, decode with the tgt,
        and project to vocabulary logits.
        """
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        logits = self.project(dec_output)
        return logits


def build_transformer(
    src_vocab_size,
    tgt_vocab_size,
    src_seq_len,
    tgt_seq_len,
    nlayers=6,
    noh=8,
    d_model=512,
    dropout=0.1,
    dff=2048,
):
    src_emb = InputEmbeddings(d_model, src_vocab_size)
    tgt_emb = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos_enc = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_enc = PositionalEncoding(d_model, tgt_seq_len, dropout)

    enc_blocks = []
    for i in range(0, nlayers):
        mha = MHA(d_model, noh, dropout)
        ff = FeedForward(d_model, dff, dropout)
        enc_block = EncoderBlock(d_model, mha, ff, dropout)
        enc_blocks.append(enc_block)

    encoder = Encoder(d_model, torch.nn.ModuleList(enc_blocks))

    dec_blocks = []
    for i in range(0, nlayers):
        mha = MHA(d_model, noh, dropout)
        mha2 = MHA(d_model, noh, dropout)
        ff = FeedForward(d_model, dff, dropout)
        dec_block = DecoderBlock(d_model, mha, mha2, ff, dropout)
        dec_blocks.append(dec_block)

    decoder = Decoder(d_model, torch.nn.ModuleList(dec_blocks))

    proj = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_pos_enc, tgt_pos_enc, src_emb, tgt_emb, proj
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return transformer
