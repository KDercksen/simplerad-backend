#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spacy
from hydra import compose
from omegaconf import DictConfig
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from ..utils import read_jsonl_dir
from .base import BaseSearcher


class SimstringJSONLFolderSearcher(BaseSearcher):
    def __init__(self, cfg: DictConfig):
        self.known_entities = read_jsonl_dir(cfg["jsonl_directory"])
        self.nlp = spacy.load(cfg["spacy_model"])
        self.db = DictDatabase(CharacterNgramFeatureExtractor(cfg["char_ngram"]))
        self.title2entity = {}
        for ent in self.known_entities:
            stemmed_title = ent["title"].lower()
            self.db.add(stemmed_title)
            self.title2entity[stemmed_title] = ent
            for s in ent.get("synonyms", []):
                stemmed_s = s.lower()
                self.db.add(stemmed_s)
                self.title2entity[stemmed_s] = ent

        self.searcher = Searcher(self.db, CosineMeasure())
        self.cosim_threshold = cfg["cosim_threshold"]

    def search(self, text: str):
        matches = self.searcher.ranked_search(text.lower(), self.cosim_threshold)
        results = []
        seen = set()
        for match in matches:
            x = {"score": match[0], "entity": self.title2entity[match[1]]}
            if (t := x["entity"]["title"]) not in seen:
                results.append(x)
                seen.add(t)
        return results
