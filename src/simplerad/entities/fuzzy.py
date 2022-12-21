from pathlib import Path
from typing import Set, Tuple

import spacy
from hydra import compose
from omegaconf import DictConfig
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from ..utils import read_jsonl
from .base import BasePredictor


class SimstringPredictor(BasePredictor):
    def __init__(self, cfg: DictConfig):
        self.nlp = spacy.load(cfg["spacy_model"], disable=["tagger", "parser"])

        self.known_entities = []
        data_path = Path(cfg["jsonl_directory"])
        for fname in data_path.glob("*.jsonl"):
            self.known_entities.extend(read_jsonl(fname))

        with open(data_path / "blacklist") as f:
            self.blacklisted = []
            for line in f:
                self.blacklisted.append(line.strip().lower())

        self.db = DictDatabase(CharacterNgramFeatureExtractor(cfg["char_ngram"]))
        for ent in self.known_entities:
            self.db.add(ent["title"].lower())
            for s in ent.get("synonyms", []):
                self.db.add(s.lower())
        self.searcher = Searcher(self.db, CosineMeasure())
        self.cosim_threshold = cfg["cosim_threshold"]
        self.word_ngram = cfg["word_ngram"]

    def predict(self, text):
        matches = []
        for sent in text["sentences"]:
            sent_start = sent["start"]
            sent_end = sent["end"]
            for start, end, ngram in self.make_ngrams(
                text["text"][sent_start:sent_end], self.word_ngram
            ):
                query = ngram.lower()
                if query in self.blacklisted:
                    continue
                results = self.searcher.ranked_search(query, self.cosim_threshold)
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
