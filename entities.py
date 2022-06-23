#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from flair.models import SequenceTagger
from flair.data import Sentence
from utils import preprocess, read_jsonl
from typing import Dict, Set, Tuple
import spacy

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher


class BasePredictor:
    def predict(self, text):
        raise NotImplementedError("subclass needs to implement this function")


class SimstringPredictor(BasePredictor):
    def __init__(self, data_path):
        self.known_entities = []
        for fname in Path(data_path).glob("*.jsonl"):
            self.known_entities.extend(read_jsonl(fname))

        self.db = DictDatabase(CharacterNgramFeatureExtractor(3))
        for ent in self.known_entities:
            self.db.add(ent["title"].lower())
        self.searcher = Searcher(self.db, CosineMeasure())

        self.nlp = spacy.load("nl_core_news_lg", disable=["tagger", "parser"])

    def predict(self, text):
        matches = []
        for sent in text["sentences"]:
            sent_start = sent["start"]
            sent_end = sent["end"]
            for start, end, ngram in self.make_ngrams(
                text["text"][sent_start:sent_end], 3
            ):
                query = ngram.lower()
                results = self.searcher.ranked_search(query, 0.7)
                if results:
                    matches.append(
                        {
                            "start": sent_start + start,
                            "end": sent_start + end,
                            "text": ngram,
                            "matches": sorted(
                                results, key=lambda x: x[0], reverse=True
                            ),
                        }
                    )
        filtered_matches = self.select_best_non_overlapping(matches)
        return [
            # do not return linked entities, that is for search.py
            {k: match[k] for k in ["start", "end", "text"]}
            for match in filtered_matches
        ]

    def make_ngrams(self, text, n, min_length=3):
        def token_allowed(tok):
            # uses spacy attributes
            return not (
                tok.is_punct or tok.is_stop or tok.text in ["+", "-", "dd", "d.d."]
            )

        doc = self.nlp(text)
        seen_ngrams = set()
        for i in range(len(doc)):
            for j in range(i + 1, min(i + n, len(doc)) + 1):
                span = doc[i:j]
                try:
                    while not token_allowed(span[0]):
                        span = span[1:]
                    while not token_allowed(span[-1]):
                        span = span[:-1]

                except IndexError:
                    # span completely emptied, skip
                    continue
                if len(span.text) < min_length:
                    continue
                val = (start := span.start_char, end := span.end_char, text[start:end])
                if val[-1] not in seen_ngrams:
                    yield val
                    seen_ngrams.add(val[-1])

    def overlaps(self, span1, span2):
        first, second = sorted([span1, span2], key=lambda x: x[0])
        return first[1] > second[0]

    def select_best_non_overlapping(self, matches):
        matches = sorted(
            matches, key=lambda x: (x["matches"][0][0], len(x["text"])), reverse=True
        )
        seen_spans: Set[Tuple[int, int]] = set()
        filtered_matches = []
        for match in matches:
            span = match["start"], match["end"]
            if not any(self.overlaps(span, x) for x in seen_spans):
                filtered_matches.append(match)
                seen_spans.add(span)
        return sorted(filtered_matches, key=lambda x: x["start"])


class FlairPredictor(BasePredictor):
    # small wrapper around Flair entity tagger

    def __init__(self, model_name):
        self.model = SequenceTagger.load(model_name)

    def predict(self, text):
        s = Sentence(text["text"])
        self.model.predict(s)

        return [
            {
                "start": (start := x.start_position),
                "end": (end := x.end_position),
                "text": text[start:end],
            }
            for x in s.get_spans("ner")
        ]


entity_taggers: Dict[str, BasePredictor] = {
    "ner-english-fast": FlairPredictor("flair/ner-english-fast"),
    "simstring": SimstringPredictor("data/entity_lists/"),
}


def get_entities(text: str, model_name: str):
    preprocessed = preprocess(text)
    preprocessed["spans"] = entity_taggers[model_name].predict(preprocessed)

    return preprocessed
