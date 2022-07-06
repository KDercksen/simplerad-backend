#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import nltk
from pathlib import Path

nltk_tokenizer = nltk.data.load("tokenizers/punkt/dutch.pickle")


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


def read_jsonl(fname):
    with open(fname) as f:
        return [json.loads(line) for line in f]


def read_jsonl_dir(path):
    items = []
    for fname in Path(path).glob("*.jsonl"):
        items.extend(read_jsonl(fname))
    return items
