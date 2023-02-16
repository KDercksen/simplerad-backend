#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
from pydantic import BaseModel


class TextClassificationResponse(BaseModel):
    class Label(BaseModel):
        value: str
        score: float

    # list of labels and probabilities
    labels: List[Label]


class SentenceClassificationResponse(BaseModel):
    class Span(BaseModel):
        start: int
        end: int
        text: str

    class Label(BaseModel):
        value: str
        score: float

    sentences: List[Span]

    # list of labels and probabilities for each sentence
    labels: List[List[Label]]
