#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
from utils import preprocess, read_jsonl
from pathlib import Path

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher


class BaseSearcher:
    def search(self, text: str):
        raise NotImplementedError("subclass should implement this function")


class SimstringJSONLFolderSearcher(BaseSearcher):
    def __init__(self, directory: Path):
        self.known_entities = []
        for filename in directory.glob("*.jsonl"):
            self.known_entities.extend(read_jsonl(filename))

        self.db = DictDatabase(CharacterNgramFeatureExtractor(3))
        self.title2entity = {}
        for ent in self.known_entities:
            self.db.add(ent["title"].lower())
            self.title2entity[ent["title"].lower()] = ent
        self.searcher = Searcher(self.db, CosineMeasure())

    def search(self, text: str):
        matches = self.searcher.ranked_search(text.lower(), 0.5)
        return [
            {"score": match[0], "entity": self.title2entity[match[1]]}
            for match in matches
        ]


searchers: Dict[str, BaseSearcher] = {
    "simstring": SimstringJSONLFolderSearcher(Path("data/entity_lists/"))
}


def get_search_results(text: str, model_name: str):
    preprocessed = preprocess(text)
    results = searchers[model_name].search(preprocessed["text"])
    return {"data": sorted(results, key=lambda x: x["score"], reverse=True)}
