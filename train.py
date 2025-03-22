from torchinfo import summary
from model import build_transformer
from util import create_resources
import yaml
import torch
from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt
from matplotlib import font_manager
import re


mangal_font_path = "Mangal.TTf"
devanagari_font = font_manager.FontProperties(fname=mangal_font_path)


class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self):
        step = max(self.step_num, 1)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        tokenizer_src,
        tokenizer_tgt,
        seq_len,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.tgt_tokenizer = tokenizer_tgt
        self.src_tokenizer = tokenizer_src
        self.seq_len = seq_len

    def train_epoch(self, dataloader):
        self.model.train()
        torch.cuda.empty_cache()
        running_loss = 0.0
        total_tokens = 0
        progress_bar = tqdm(
            enumerate(dataloader), desc="Training", total=len(dataloader)
        )

        for batch_idx, batch in progress_bar:

            encoder_input = batch["encoder_input"].to(self.device)
            # Should be (1,seq_len) => (batch_size,seq_len)
            decoder_input = batch["decoder_input"].to(self.device)
            # Should be (1,seq_len) => (batch_size,seq_len)

            encoder_mask = batch["encoder_mask"].to(self.device)
            decoder_mask = batch["decoder_mask"].to(self.device)

            encoder_output = self.model.encode(encoder_input, encoder_mask)
            decoder_output = self.model.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )
            projection_output = self.model.project(decoder_output)

            label = batch["label"].to(self.device)

            loss = self.criterion(
                projection_output.view(-1, self.tgt_tokenizer.get_vocab_size()),
                label.view(-1),
            )

            loss.backward()

            self.optimizer.step()
            current_lr = self.scheduler.step()
            self.optimizer.zero_grad()

            pad_id = 1
            with torch.no_grad():
                non_pad = label.ne(pad_id)
                num_nonpad_tokens = non_pad.sum().item()
                running_loss += loss.item() * num_nonpad_tokens
                total_tokens += num_nonpad_tokens

            if (batch_idx + 1) % 50 == 0:
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "learning_rate": current_lr,
                        "batch": batch_idx + 1,
                    }
                )

        epoch_loss = running_loss / total_tokens if total_tokens > 0 else 0.0
        return epoch_loss

    def save_checkpoint(self, epoch, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.step_num,
        }
        torch.save(checkpoint, os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
        print(f"Checkpoint saved at epoch {epoch}")

    def run(self, train_loader, epochs, output_dir, start_epoch=1):
        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            current_lr = self.scheduler.get_lr()

            wandb.log(
                {"epoch": epoch, "train_loss": train_loss, "learning_rate": current_lr}
            )

            self.save_checkpoint(epoch, output_dir)


def load_latest_checkpoint(model, optimizer, scheduler, model_directory, device):
    if not os.path.isdir(model_directory):
        return None, 1
    checkpoint_files = []
    for filename in os.listdir(model_directory):
        if filename.endswith(".pth"):
            match = re.search(r"model_epoch_(\d+)\.pth", filename)
            if match:
                epoch = int(match.group(1))
                checkpoint_files.append((epoch, filename))

    if not checkpoint_files:
        return None, 1

    # Get the checkpoint with the highest epoch number
    latest_epoch, latest_filename = max(checkpoint_files, key=lambda x: x[0])
    ckpt_path = os.path.join(model_directory, latest_filename)
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.step_num = ckpt["scheduler_state"]
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming Training from epoch {ckpt['epoch']}")
    return ckpt, start_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    ) = create_resources()
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    run = wandb.init(project="AttentionTranslate", config=config)

    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        config["num_enc_dec_blocks"],
        config["num_of_heads"],
        config["d_model"],
    )

    model = model.to(device)

    wandb.watch(model, log="all")

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = NoamScheduler(optimizer, config["d_model"], config["warmup_steps"])

    start_epoch = 1

    if config["resume_training"]:
        ckpt, start_epoch = load_latest_checkpoint(
            model, optimizer, scheduler, config["model_directory"], device
        )

    if start_epoch == 1:
        print("Training from scratch.")

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
    )

    # test_data_subset = list(test_dataloader)
    # one_percent = int(0.01 * len(test_data_subset))
    # test_data_1_percent = test_data_subset[:one_percent]

    trainer.run(
        train_dataloader, config["epochs"], config["model_directory"], start_epoch
    )

    run.finish()


if __name__ == "__main__":
    main()
