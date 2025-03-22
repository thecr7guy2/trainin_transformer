from pathlib import Path
from model import build_transformer
from util import create_resources
import torch
import sys
import yaml

def translate(sentence: str):

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_dataloader,valid_dataloader,test_dataloader,tokenizer_src,tokenizer_tgt = create_resources()

    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()


    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        config["num_enc_dec_blocks"],
        config["num_of_heads"],
        config["d_model"]
    )

    model = model.to(device)
    model_filename = "models/model.pth"
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        print(source,source.ids)

        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config["seq_len"] - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0)

        source = source.to(device)
        source = source.unsqueeze(0)

        print(source.shape)



        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        print(source_mask)

        encoder_output = model.encode(source, source_mask)
        print(encoder_output.shape)
        decoder_input = torch.full((1, 1), tokenizer_tgt.token_to_id('[SOS]'),
                                dtype=torch.long, device=device)

       
        while decoder_input.size(1) < config["seq_len"]:

            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                                    diagonal=1).to(device, dtype=torch.int)
            
            print("#######################")

            
            print(decoder_mask.shape)
            out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
            prob = model.project(out[:, -1])
            print(max(prob[0]))
            print(min(prob[0]))
            _, next_word = torch.max(prob, dim=1)
            
            next_token = torch.full((1, 1), next_word.item(), dtype=torch.long, device=device)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')
            
            if next_word.item() == tokenizer_tgt.token_to_id('[EOS]'):
                break


    # return tokenizer_tgt.decode(decoder_input[0].tolist())
    return 0
    

a = translate("My Name is sai and I love computers")
print(a)