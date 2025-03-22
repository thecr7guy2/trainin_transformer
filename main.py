from torchinfo import summary
from model import build_transformer
from util import create_resources
import yaml
import torch
from pathlib import Path





with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


train_dataloader,valid_dataloader,test_dataloader,tokenizer_src,tokenizer_tgt = create_resources()
src_vocab_size = tokenizer_src.get_vocab_size()
tgt_vocab_size = tokenizer_src.get_vocab_size()

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
num_epochs = config["epochs"] if "epochs" in config else 10


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

criterion = loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],eps=1e-9)


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from epoch {start_epoch}")
    return start_epoch


def train_one_epoch(device):
    model.train()
    running_loss = 0.0




def train_model(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader,valid_dataloader,test_dataloader,tokenizer_src,tokenizer_tgt = create_resources()

    


    




