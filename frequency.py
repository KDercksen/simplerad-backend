#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from hydra import compose
from utils import LazyValueDict, read_json, preprocess, lemmatize_tokens
import spacy


class BaseFrequency:
    def get_frequency(self, term: str):
        raise NotImplementedError("subclass should implement this function")


class SimstringGeneratedFrequencies(BaseFrequency):
    def __init__(self):
        cfg = compose(config_name="config")["frequency"]
        data_path = Path(cfg["json_path"])
        self.frequencies = read_json(data_path)
        self.nlp = spacy.load(cfg["spacy_model"])

    def get_frequency(self, term: str):
        return self.frequencies.get(lemmatize_tokens(term, self.nlp), 0.0)


frequencizers = LazyValueDict(
    {
        "simstring": SimstringGeneratedFrequencies,
    }
)


def get_frequencies(text: str, model_name: str):
    preprocessed = preprocess(text)
    freq = frequencizers[model_name].get_frequency(preprocessed["text"])
    return {"frequency": freq}
