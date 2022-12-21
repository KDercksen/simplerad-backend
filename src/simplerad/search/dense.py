#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import faiss
from gensim.models import FastText
from hydra import compose
from omegaconf import DictConfig

from ..utils import read_jsonl_dir, simple_tokenize
from .base import BaseSearcher


class FastTextFAISSJSONLFolderSearcher(BaseSearcher):
    def __init__(self, cfg: DictConfig):
        self.known_entities = read_jsonl_dir(cfg["jsonl_directory"])
        self.model = FastText.load(cfg["fasttext_path"])
        self.index = faiss.IndexFlatL2(self.model.vector_size)

        # initialize index
        # TODO: support for synonyms?
        for ent in self.known_entities:
            tok = simple_tokenize(ent["title"])
            if len(tok) == 0:
                continue
            vector = self.model.wv.get_sentence_vector(tok)[None, ...]
            self.index.add(vector)

        self.top_n = cfg["top_n"]

    def search(self, text: str):
        vector = self.model.wv.get_sentence_vector(simple_tokenize(text))[None, ...]
        distances, indexes = self.index.search(vector, self.top_n)
        distances, indexes = distances[0].tolist(), indexes[0].tolist()
        return [
            {"score": 1 - d, "entity": self.known_entities[i]}
            for d, i in sorted(zip(distances, indexes))
        ]
