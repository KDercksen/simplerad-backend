#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from pydantic import BaseModel


class EntityTaggerResponse(BaseModel):
    class Span(BaseModel):
        start: int
        end: int
        text: str

    text: str
    sentences: List[Span]
    spans: List[Span]
