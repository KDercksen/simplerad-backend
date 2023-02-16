#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..utils import LazyValueDict, preprocess
from .sentence_classification import FlairSentenceClassifier
from .text_classification import FlairTextClassifier

text_classifiers = LazyValueDict(
    {
        "flair": FlairTextClassifier,
    }
)

sentence_classifiers = LazyValueDict(
    {
        "flair": FlairSentenceClassifier,
    }
)


def get_sentence_classification(text: str):
    preprocessed = preprocess(text)
    labels = sentence_classifiers.get_model().predict(preprocessed)
    return {"labels": labels, "sentences": preprocessed["sentences"]}


def get_text_classification(text: str):
    preprocessed = preprocess(text)
    label = text_classifiers.get_model().predict(preprocessed)
    return {"labels": label}


__all__ = [
    text_classifiers,
    sentence_classifiers,
    get_sentence_classification,
    get_text_classification,
]
