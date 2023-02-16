#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import BaseTextClassifier

from flair.models import TextClassifier
from flair.data import Sentence
from omegaconf import DictConfig


class FlairTextClassifier(BaseTextClassifier):
    def __init__(self, cfg: DictConfig):
        self.model = TextClassifier.load(cfg["model_name"])

    def predict(self, text):
        s = Sentence(text["text"])
        self.model.predict(s, return_probabilities_for_all_classes=True)
        return [
            {"value": x.value, "score": x.score} for x in s.labels[0].data_point.labels
        ]
