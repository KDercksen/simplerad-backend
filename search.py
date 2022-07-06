#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import faiss
from gensim.models import FastText
from gensim.utils import simple_preprocess
from nltk.stem.snowball import DutchStemmer
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from utils import preprocess, read_jsonl_dir


class BaseSearcher:
    def search(self, text: str):
        raise NotImplementedError("subclass should implement this function")


class SimstringJSONLFolderSearcher(BaseSearcher):
    def __init__(self, jsonl_directory):
        self.known_entities = read_jsonl_dir(jsonl_directory)
        self.stemmer = DutchStemmer()
        self.db = DictDatabase(CharacterNgramFeatureExtractor(3))
        self.title2entity = {}
        for ent in self.known_entities:
            stemmed_title = self.stemmer.stem(ent["title"].lower())
            self.db.add(stemmed_title)
            self.title2entity[stemmed_title] = ent
        self.searcher = Searcher(self.db, CosineMeasure())

    def search(self, text: str):
        matches = self.searcher.ranked_search(text.lower(), 0.5)
        return [
            {"score": match[0], "entity": self.title2entity[match[1]]}
            for match in matches
        ]


class FastTextFAISSJSONLFolderSearcher(BaseSearcher):
    def __init__(self, jsonl_directory, embedding_model):
        self.known_entities = read_jsonl_dir(jsonl_directory)
        self.model = FastText.load(embedding_model)
        self.index = faiss.IndexFlatL2(self.model.vector_size)

        # initialize index
        for ent in self.known_entities:
            vector = self.model.wv.get_sentence_vector(
                simple_preprocess(ent["title"], max_len=100)
            )[None, ...]
            self.index.add(vector)

    def search(self, text: str):
        vector = self.model.wv.get_sentence_vector(
            simple_preprocess(text, max_len=100)
        )[None, ...]
        distances, indexes = self.index.search(vector, 10)
        distances, indexes = distances[0].tolist(), indexes[0].tolist()
        return [
            {"score": d, "entity": self.known_entities[i]}
            for d, i in sorted(zip(distances, indexes), reverse=True)
        ]


searchers: Dict[str, BaseSearcher] = {
    "simstring": SimstringJSONLFolderSearcher("data/entity_lists/"),
    "faiss": FastTextFAISSJSONLFolderSearcher(
        "data/entity_lists/", "models/fasttext_cbow_300_5epochs.bin"
    ),
}


def get_search_results(text: str, model_name: str):
    preprocessed = preprocess(text)
    results = searchers[model_name].search(preprocessed["text"])
    return {"data": sorted(results, key=lambda x: x["score"], reverse=True)}
