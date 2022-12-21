#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from collections import defaultdict
from functools import reduce

import spacy
from hydra import compose
from omegaconf import DictConfig

from ..utils import read_jsonl_dir, simple_tokenize
from .base import BaseTwoStageSearcher


class BaseInvertedIndex(BaseTwoStageSearcher):
    def __init__(self, nlp, jsonl_directory, fields):
        self.known_entities = read_jsonl_dir(jsonl_directory)
        self.nlp = spacy.load(nlp)

        # build inverted index on title, description fields
        # TODO: add support for synonyms here as well?
        self.index = defaultdict(set)
        total_len = 0
        for i, ent in enumerate(self.known_entities):
            for field in fields:
                content = simple_tokenize(ent[field])
                total_len += len(content)
                for term in content:
                    self.index[term].add(i)
        self.avg_doc_len = total_len / len(self.known_entities)

    def search(self, text: str):
        query_terms = simple_tokenize(text)
        if len(query_terms) == 0:
            return []

        ent_indexes = reduce(
            lambda x, y: x | y, [self.index[term] for term in query_terms]
        )
        scores = self.rank(query_terms, ent_indexes)
        return [
            {"score": score, "entity": self.known_entities[idx]}
            for (score, idx) in sorted(zip(scores, ent_indexes), reverse=True)
        ]


class BM25(BaseInvertedIndex):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg["spacy_model"], cfg["jsonl_directory"], cfg["fields"])

        self.b = cfg["b"]
        self.k1 = cfg["k1"]

    def rank(self, query_terms, indexes):
        # query_terms is already preprocessed
        entities = [self.known_entities[i] for i in indexes]
        entities = [
            simple_tokenize(
                "\n".join(ent.get("synonyms", []) + [ent["title"], ent["description"]])
            )
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
