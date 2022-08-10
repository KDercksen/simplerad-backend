#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import defaultdict
from functools import reduce

import faiss
from gensim.models import FastText
from hydra import compose
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher
import spacy

from utils import (
    LazyValueDict,
    preprocess,
    read_jsonl_dir,
    simple_tokenize,
    lemmatize_tokens,
)


class BaseSearcher:
    def search(self, text: str):
        raise NotImplementedError("subclass should implement this function")


class BaseTwoStageSearcher(BaseSearcher):
    def rank(self, query_terms, indexes):
        raise NotImplementedError("subclass should implement this function")


class BaseInvertedIndex(BaseTwoStageSearcher):
    def __init__(self, nlp, jsonl_directory, fields):
        self.known_entities = read_jsonl_dir(jsonl_directory)
        self.nlp = spacy.load(nlp)

        # build inverted index on title, description fields
        self.index = defaultdict(set)
        total_len = 0
        for i, ent in enumerate(self.known_entities):
            for field in fields:
                content = simple_tokenize(ent[field])
                total_len += len(content)
                for term in content:
                    self.index[lemmatize_tokens(term, self.nlp)].add(i)
        self.avg_doc_len = total_len / len(self.known_entities)

    def search(self, text: str):
        query_terms = [
            lemmatize_tokens(term, self.nlp) for term in simple_tokenize(text)
        ]
        ent_indexes = reduce(
            lambda x, y: x | y, [self.index[term] for term in query_terms]
        )
        scores = self.rank(query_terms, ent_indexes)
        return [
            {"score": score, "entity": self.known_entities[idx]}
            for (score, idx) in sorted(zip(scores, ent_indexes), reverse=True)
        ]


class BM25(BaseInvertedIndex):
    def __init__(self):
        cfg = compose(config_name="config")["search"]

        super().__init__(cfg["spacy_model"], cfg["jsonl_directory"], cfg["fields"])

        self.b = cfg["b"]
        self.k1 = cfg["k1"]

    def rank(self, query_terms, indexes):
        # query_terms is already preprocessed
        entities = [self.known_entities[i] for i in indexes]
        entities = [
            [
                lemmatize_tokens(term, self.nlp)
                for term in simple_tokenize(
                    "\n".join([ent["title"], ent["description"]])
                )
            ]
            for ent in entities
        ]

        scores = []
        for i, ent in zip(indexes, entities):
            score = 0.0
            for term in query_terms:
                idf = math.log(
                    (
                        (len(self.known_entities) - len(self.index[term]) + 0.5)
                        / (len(self.index[term]) + 0.5)
                    )
                    + 1
                )
                tf = ent.count(term)
                tmp = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * (len(ent) / self.avg_doc_len))
                )
                score += idf * tmp
            scores.append(score)
        return scores


class SimstringJSONLFolderSearcher(BaseSearcher):
    def __init__(self):
        cfg = compose(config_name="config")["search"]

        self.known_entities = read_jsonl_dir(cfg["jsonl_directory"])
        self.nlp = spacy.load(cfg["spacy_model"])
        self.db = DictDatabase(CharacterNgramFeatureExtractor(cfg["char_ngram"]))
        self.title2entity = {}
        for ent in self.known_entities:
            stemmed_title = lemmatize_tokens(ent["title"].lower(), self.nlp)
            self.db.add(stemmed_title)
            self.title2entity[stemmed_title] = ent
        self.searcher = Searcher(self.db, CosineMeasure())
        self.cosim_threshold = cfg["cosim_threshold"]

    def search(self, text: str):
        matches = self.searcher.ranked_search(text.lower(), self.cosim_threshold)
        return [
            {"score": match[0], "entity": self.title2entity[match[1]]}
            for match in matches
        ]


class FastTextFAISSJSONLFolderSearcher(BaseSearcher):
    def __init__(self):
        cfg = compose(config_name="config")["search"]

        self.known_entities = read_jsonl_dir(cfg["jsonl_directory"])
        self.model = FastText.load(cfg["fasttext_path"])
        self.index = faiss.IndexFlatL2(self.model.vector_size)

        # initialize index
        for ent in self.known_entities:
            vector = self.model.wv.get_sentence_vector(simple_tokenize(ent["title"]))[
                None, ...
            ]
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


searchers = LazyValueDict(
    {
        "simstring": SimstringJSONLFolderSearcher,
        "faiss": FastTextFAISSJSONLFolderSearcher,
        "bm25": BM25,
    }
)


def get_search_results(text: str, model_name: str):
    preprocessed = preprocess(text)
    results = searchers[model_name].search(preprocessed["text"])
    return {"data": sorted(results, key=lambda x: x["score"], reverse=True)}
