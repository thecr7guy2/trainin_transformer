from torchinfo import summary
from model import build_transformer
from util import create_resources
import yaml
import torch



train_dataloader,valid_dataloader,test_dataloader,tokenizer_src,tokenizer_tgt = create_resources()
src_vocab_size = tokenizer_src.get_vocab_size()
tgt_vocab_size = tokenizer_src.get_vocab_size()


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model = build_transformer(
    src_vocab_size,
    tgt_vocab_size,
    config["seq_len"],
    config["seq_len"],
    config["num_enc_dec_blocks"],
    config["num_of_heads"],
    config["d_model"]
)

batch_size = config["batch_size"]


# # Fake token IDs
src = torch.randint(0, 30000, (batch_size, config["seq_len"]), dtype=torch.int64)
tgt = torch.randint(0, 30000, (batch_size, config["seq_len"]), dtype=torch.int64)

# # Example binary masks (1=keep, 0=mask out); shape [batch_size, 1, seq_len, seq_len]
src_mask = torch.ones(batch_size, 1, config["seq_len"], config["seq_len"], dtype=torch.bool)
tgt_mask = torch.ones(batch_size, 1, config["seq_len"], config["seq_len"], dtype=torch.bool)


summary(
    model,
    input_data=(src, tgt, src_mask, tgt_mask),
    col_names=("input_size", "output_size", "num_params", "trainable"),
    depth=4
)