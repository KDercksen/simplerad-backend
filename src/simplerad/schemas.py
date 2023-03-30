#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from pydantic import BaseModel

from .entities.schemas import EntityTaggerResponse
from .prevalence.schemas import PrevalenceResponse
from .search.schemas import SearchResponse
from .summarization.schemas import SummaryResponse
from .classification.schemas import (
    TextClassificationResponse,
    SentenceClassificationResponse,
)


class TextRequest(BaseModel):
    text: str


__all__ = [
    EntityTaggerResponse,
    PrevalenceResponse,
    SearchResponse,
    SummaryResponse,
    TextRequest,
    TextClassificationResponse,
    SentenceClassificationResponse,
]
