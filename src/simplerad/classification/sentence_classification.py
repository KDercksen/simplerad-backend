#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from .base import BaseSentenceClassifier
from flair.models import TextClassifier
from flair.data import Sentence
from omegaconf import DictConfig


class FlairSentenceClassifier(BaseSentenceClassifier):
    def __init__(self, cfg: DictConfig):
        self.model = TextClassifier.load(cfg["model_name"])

    def predict(self, text):
        s = [Sentence(x["text"]) for x in text["sentences"]]
        self.model.predict(s, return_probabilities_for_all_classes=True)

        return [
            [
                {"value": x.value, "score": x.score}
                for x in sent.labels[0].data_point.labels
            ]
            for sent in s
        ]
