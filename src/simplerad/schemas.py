#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from pydantic import BaseModel

from .entities.schemas import EntityTaggerResponse
from .frequency.schemas import FrequencyResponse
from .search.schemas import SearchResponse
from .summarization.schemas import SummaryResponse


class TextRequest(BaseModel):
    text: str


__all__ = [
    EntityTaggerResponse,
    FrequencyResponse,
    SearchResponse,
    SummaryResponse,
    TextRequest,
]
