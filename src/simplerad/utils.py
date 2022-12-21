#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import UserDict
from pathlib import Path

import nltk
from gensim.utils import simple_preprocess as sp
from omegaconf import DictConfig

nltk_tokenizer = nltk.data.load("tokenizers/punkt/dutch.pickle")


def simple_tokenize(text):
    return sp(text, max_len=100)


def preprocess(text: str):
    normalized = re.sub(r"\s+", " ", text)

    if len(normalized) == 0:
        # empty input?
        return {"text": "", "sentences": []}

    sents = nltk_tokenizer.tokenize(normalized, realign_boundaries=True)
    bounds = [(0, len(sents[0]))]

    for i, sent in enumerate(sents[1:], start=1):
        start = bounds[i - 1][1] + 1  # account for whitespace between sentences
        bounds.append((start, start + len(sent)))
    return {
        "text": "\n".join(sents),
        "sentences": [
            {"start": start, "end": end, "text": sent}
            for ((start, end), sent) in zip(bounds, sents)
        ],
    }


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def read_jsonl(fname):
    with open(fname) as f:
        return [json.loads(line) for line in f]


def read_jsonl_dir(path):
    items = []
    for fname in Path(path).glob("*.jsonl"):
        items.extend(read_jsonl(fname))
    return items


class LazyValueDict:
    def __init__(self, initial_data):
        self.data = initial_data
        self.hotkey = None
        self.value = None
        self.config = None

    def set_config(self, config: DictConfig):
        self.config = config

    def get_model(self):
        if self.config is None:
            raise ValueError(
                "first call set_config before trying to initialize/access lazy values"
            )
        key = self.config["name"]
        if key == self.hotkey and self.value:
            # already loaded model
            return self.value
        # cache different model
        self.hotkey = key
        self.value = self.data[key](self.config)
        return self.value
