import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset
import json


class English2HindiDataset(Dataset):
    def __init__(
        self, data, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
     
        self.data = data
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

        # Why int64?
        # When passing token indices to an nn.Embedding layer, PyTorch expects torch.int64 (or torch.long)

    def __len__(self):
        return len(self.data)
    
    def causal_mask(self,size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    def __getitem__(self, idx):
        trans_pairs = self.data[idx]

        src_text = trans_pairs["en_text"]
        tgt_text = trans_pairs["hi_text"]
        # here we first get english_text and hindi text which was in dictionary.

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # we then use the two tokenizers to encode the text into input_ids.

        # For every sentence for example : I am sai - The input to the enocoder will be
        ## <SOS> I am Sai <EOS>
        ### but because of variable length sequences we need to add padding.
        ### how do we do it ?  we take the longest sentence in the dataset add 30 to it and that will gives us the seq_len
        ### So now we add padding to every sentence and make it similar lengths.

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2

        # -2 Because of the <SOS> and <EOS>

        dec_num_padding_tokens = self.seq_len = len(dec_input_tokens) - 1

        # since the input to the decoder will only consist of <SOS>
        # the target label will have the <EOS>

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # why dim =0 because everything 1d so they are stacked one after other.

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Label is always tgt language so we give the the decoder tokens

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
        decoder_mask= (decoder_input != self.pad_token).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),

        ### didnt understand this at all.


        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input,  
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }



class English2HindiDatasetTest(Dataset):
    def __init__(
        self, json_path,
    ):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    


    def __getitem__(self, idx):
        trans_pairs = self.data[idx]
        return trans_pairs