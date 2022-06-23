#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
    model_name: str


class EntityTaggerResponse(BaseModel):
    class Span(BaseModel):
        start: int
        end: int
        text: str

    text: str
    sentences: List[Span]
    spans: List[Span]


class SummaryResponse(BaseModel):
    summary: str


class SearchEntity(BaseModel):
    title: str
    description: str
    url: str
    source_id: str
    source: str


class SearchResponse(BaseModel):
    class SingleSearchResult(BaseModel):
        score: float
        entity: SearchEntity

    data: List[SingleSearchResult]
