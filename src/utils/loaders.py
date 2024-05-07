import json
import pandas as pd
from pathlib import Path

import lzma

from parsivar import Normalizer
import torch


def load_mewsli(tsv_path: Path, lowercase=False) -> tuple[list[str], list[str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    if lowercase:
        df["mention"] = df["mention"].str.lower()
    # if "fa" in tsv_path.parts:
    #     normalizer = Normalizer()
    #     print("normalizing farsi")
    #     df["mention"] = df["mention"].apply(normalizer.normalize)

    return df["mention"].tolist(), df["qid"].apply(lambda x: int(x[1:])).tolist()


def load_damuel(
    dir_path: Path, only_wiki: bool, use_xz=False, lowercase=False
) -> tuple[list[str], list[str]]:
    def process_line(line):
        loaded = json.loads(line)
        if "wiki" not in loaded:
            return
        wiki = loaded["wiki"]
        tokens = wiki["tokens"]
        text = wiki["text"]
        if lowercase:
            text = text.lower()
        # if "damuel_1.0_fa" in dir_path.parts:
        #     normalizer = Normalizer()
        #     text = normalizer.normalize(text)
        for l in wiki["links"]:
            if "qid" not in l:
                continue
            if only_wiki and l["origin"] != "wiki":
                continue
            start = l["start"]
            end = l["end"] - 1
            try:
                mention_slice = slice(tokens[start]["start"], tokens[end]["end"])
            except IndexError:
                print(start, end, len(tokens))
            mention_names.append(text[mention_slice])
            qids.append(int(l["qid"][1:]))

    # if "damuel_1.0_fa" in dir_path.parts:
    #     print("normalizing farsi")

    mention_names, qids = [], []
    for fn in dir_path.iterdir():
        print("processing", fn)
        if use_xz:
            with lzma.open(fn, "r") as f:
                for line in f:
                    process_line(line)
        else:
            for line in fn.open():
                process_line(line)
        print("processed", fn)
    return mention_names, qids


def load_mewsli_from_files(tsv_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    mentions, qids = [], []
    for row in df.itertuples():
        with open(tsv_path.parent / "text" / row.docid, "r") as f:
            text = f.read()
            mentions.append(text[row.position : row.position + row.length])
            qids.append(int(row.qid[1:]))
    return mentions, qids


def load_damuel_context(
    dir_path: Path, only_wiki: bool, context_char_size
) -> tuple[list[str], list[str]]:
    contexts, qids = [], []
    for fn in dir_path.iterdir():
        for line in fn.open():
            loaded = json.loads(line)
            if "wiki" not in loaded:
                continue
            wiki = loaded["wiki"]
            tokens = wiki["tokens"]
            text = wiki["text"]
            for l in wiki["links"]:
                if "qid" not in l:
                    continue
                if only_wiki and l["origin"] != "wiki":
                    continue
                start = l["start"]
                end = l["end"] - 1
                # assert start >= 0
                # assert end < len(tokens)
                # print(start, end, len(tokens))
                char_start = tokens[start]["start"]
                char_end = tokens[end]["end"]
                char_start = max(0, char_start - context_char_size)
                char_end = min(len(text), char_end + context_char_size)
                contexts.append(text[char_start:char_end])
                qids.append(int(l["qid"][1:]))
        print("processed", fn)
    return contexts, qids


def load_mewsli_context_from_files(
    tsv_path: Path, context_char_size
) -> tuple[list[str], list[str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    contexts, qids = [], []
    for row in df.itertuples():
        with open(tsv_path.parent / "text" / row.docid, "r") as f:
            text = f.read()
            char_start = max(0, row.position - context_char_size)
            char_end = min(len(text), row.position + row.length + context_char_size)
            contexts.append(text[char_start:char_end])
            qids.append(int(row.qid[1:]))
    return contexts, qids


def get_emb_state_dict(model_state_dict_path):
    """Loads a state dict from a file and returns it.

    Handles various conversions.
    This is needed because for older models we store just the dict of the underlying embedding model.
    In some newer models that train parameters above the embedding model, we store the whole state dict.
    Here we just return the model part of the state dict which is all that is needed to construct the embeddings.
    """
    d = torch.load(model_state_dict_path)
    if "softmax_multiplier" in d:
        return {k.replace("model.", ""): v for k, v in d.items() if "model." in k}
    return d
