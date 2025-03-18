from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    WhitespaceSplit,
    Punctuation,
    Sequence as PreSequence,
)
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from pathlib import Path
import yaml
from datasets import load_dataset
from dataset import English2HindiDataset, English2HindiDatasetTest
from torch.utils.data import DataLoader
import json
import random
from sklearn.model_selection import train_test_split


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]


def get_or_build_tokenizer(tokenizer_path, ds, lang):
    tokenizer_path = Path(tokenizer_path)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        if lang == "en_text":
            tokenizer.normalizer = Sequence(
                [
                    NFD(),
                    StripAccents(),
                    Lowercase(),
                ]
            )
        else:
            tokenizer.normalizer = Sequence(
                [
                    NFD(),
                    Lowercase(),
                ]
            )

        tokenizer.pre_tokenizer = PreSequence([WhitespaceSplit(), Punctuation()])
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=3,
            vocab_size=60000,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def preprocess_to_json(dataset_hf, output_json_path, word_limit=77, char_limit=300):
    filtered_data = []

    for row in dataset_hf:
        if not row.get("translated", False):
            continue

        en_value = None
        en_conversations = row.get("en_conversations", [])
        for conv in en_conversations:
            if "human" in conv["from"]:
                en_value = conv["value"]
                break
        if en_value is None and len(en_conversations) > 0:
            en_value = en_conversations[0]["value"]

        hi_value = None
        hi_conversations = row.get("conversations", [])
        for conv in hi_conversations:
            if "human" in conv["from"]:
                hi_value = conv["value"]
                break
        if hi_value is None and len(hi_conversations) > 0:
            hi_value = hi_conversations[0]["value"]

        if en_value and hi_value:
            if (
                len(en_value.split()) > word_limit
                or len(en_value) > char_limit
                or len(hi_value.split()) > word_limit
                or len(hi_value) > char_limit
            ):
                continue

            filtered_data.append({"en_text": en_value, "hi_text": hi_value})

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)



def create_resources():
    config_path = "config.yaml"
    config = load_config(config_path)

    dataset_json_path = config.get("dataset_path", "data/english2hindi_data.json")

    if not Path(dataset_json_path).exists():
        print(f"Dataset file {dataset_json_path} not found. Creating it...")
        dataset_hf = load_dataset("BhabhaAI/openhermes-2.5-hindi", split="train")
        preprocess_to_json(dataset_hf, dataset_json_path)
    else:
        print(f"Dataset file {dataset_json_path} already exists. Skipping preprocessing.")


    with open(dataset_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)



    tokenizer_src = get_or_build_tokenizer(
        config["src_tokenizer_file"], raw_data, config["src_lang"]
    )
    tokenizer_tgt = get_or_build_tokenizer(
        config["tgt_tokenizer_file"], raw_data, config["tgt_lang"]
    )


    test_pt_dataset = English2HindiDatasetTest(config["dataset_path"])

    print(len(test_pt_dataset))


    #######################################################
    ################ Sanity check #########################
    sentence = "Hello, my name is Sai !"
    encoded_sentence = tokenizer_src.encode(sentence)
    print("Input IDs:", encoded_sentence.ids)

    tokens = encoded_sentence.tokens
    print("Tokens:", tokens)


    sentence = "नमस्ते, मेरा नाम साईं है !"
    encoded_sentence = tokenizer_tgt.encode(sentence)
    print("Input IDs:", encoded_sentence.ids)

    tokens = encoded_sentence.tokens
    print("Tokens:", tokens)
    ######################################################
    ######################################################


    max_len_src = 0
    max_len_tgt = 0


    seq_len = config["seq_len"]

    if seq_len == 0:
        print("seq_len is 0, starting process...")

        
        for item in raw_data:
            src_ids = tokenizer_src.encode(item[config['src_lang']]).ids
            tgt_ids = tokenizer_tgt.encode(item[config['tgt_lang']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            
            max_len_tgt = max(max_len_tgt, len(tgt_ids))


        print(max_len_src,max_len_tgt )  

        final_max_len = max(max_len_src, max_len_tgt) + 30

        config['seq_len'] = final_max_len


        with open("config.yaml", 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        print(f'Updated seq_len to {final_max_len}')

    else:
        print("seq_len is not 0, skipping process.")

    random.seed(42)

    train_data, temp_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    test_data, valid_data = train_test_split(temp_data, test_size=0.5, random_state=42)


    print("######################################################")

    train_dataset = English2HindiDataset(
        train_data,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )

    valid_dataset = English2HindiDataset(
        valid_data,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )

    test_dataset = English2HindiDataset(
        test_data,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )


    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader,valid_dataloader,test_dataloader,tokenizer_src,tokenizer_tgt







